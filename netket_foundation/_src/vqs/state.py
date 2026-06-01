from typing import Any
import os
import warnings
from functools import partial
from collections.abc import Callable

import numpy as np

import jax
from jax import numpy as jnp

from flax import linen as nn, core as fcore
from flax.core.scope import CollectionFilter, DenyList  # noqa: F401

from netket import jax as nkjax
from netket.jax import sharding
from netket import config
from netket.sampler import (
    Sampler,
    SamplerState,
    MetropolisSampler,
    rules,
    ParallelTemperingSampler,
)
from netket.stats import Stats, statistics, LocalEstimators
from netket.hilbert import AbstractHilbert
from netket.vqs import (
    VariationalState,
    MCState,
    expect,
    get_local_kernel,
    get_local_kernel_arguments,
    local_estimators as nk_local_estimators,
)
from netket.utils import dispatch
from netket.utils.types import PyTree, SeedT
from netket.utils import (
    model_frameworks,
    wrap_to_support_scalar,
    timing,
)

from netket.vqs.mc import (
    check_hilbert,
)
from netket.vqs.mc.mc_state.state import (
    compute_chain_length,
    _is_power_of_two,
    jit_evaluate,
)

from nqxpack import save as _save, load as _load

from netket_foundation._src.operator.parametrized import (
    ParametrizedOperator,
)  # noqa: F401

from jax.sharding import NamedSharding, PartitionSpec as P
from netket_foundation._src.hilbert.parameter_space import ParameterSpace
from netket_foundation._src.nn.instance_wrapper import FoundationalInstance


def wrap_sampler(sampler, parameter_space, joint_space):
    joint_rule = rules.TensorRule(joint_space, rules=(sampler.rule, rules.FixedRule()))

    if isinstance(sampler, ParallelTemperingSampler):
        return ParallelTemperingSampler(
            joint_space,
            joint_rule,
            n_replicas=sampler.n_replicas,
            betas=sampler._beta_distribution,
            sweep_size=sampler.sweep_size,
            n_chains=sampler.n_chains,
            chunk_size=sampler.chunk_size,
            machine_pow=sampler.machine_pow,
            dtype=float,  # O sampler.dtype
        )

    elif isinstance(sampler, MetropolisSampler):
        return MetropolisSampler(
            joint_space,
            joint_rule,
            sweep_size=sampler.sweep_size,
            reset_chains=sampler.reset_chains,
            n_chains=sampler.n_chains,
            chunk_size=sampler.chunk_size,
            machine_pow=sampler.machine_pow,
            dtype=float,  # O sampler.dtype
        )

    else:
        raise NotImplementedError(
            "Il wrap_sampler supporta solo Metropolis o ParallelTemperingSampler."
        )


class FoundationalQuantumState(VariationalState):
    """Variational state that jointly samples physical configurations and fixed parameter replicas."""

    _sampler: Sampler
    """The sampler used to sample the Hilbert space."""
    sampler_state: SamplerState
    """The current state of the sampler."""
    _previous_sampler_state: SamplerState | None = None
    """The sampler state before the last sampling has been effected.

    This field is used so that we don't need to serialize the current samples
    but we can always regenerate them.
    """

    #############
    #  Settings #
    #############
    _chain_length: int = 0
    """Length of the Markov chain used for sampling configurations."""
    _n_discard_per_chain: int = 0
    """Number of samples discarded at the beginning of every Markov chain."""
    _chunk_size: int | None = None
    """The chunk size used in the evaluation of the model."""

    def __init__(
        self,
        sampler: Sampler,
        model: Any,
        parameter_space: ParameterSpace,
        n_samples: int = 1024,
        seed: SeedT | None = None,
        sampler_seed: SeedT | None = None,
        n_discard_per_chain: int | None = None,
        chunk_size: int | None = None,
        n_replicas: int | None = None,
    ):
        ###
        variables = None
        ###
        self.mutable = False
        self.training_kwargs = fcore.freeze({})
        ###
        self.parameter_space = parameter_space
        self._physical_sampler = sampler

        hilbert_physical = sampler.hilbert
        joint_space = hilbert_physical * parameter_space

        if not isinstance(sampler, MetropolisSampler | ParallelTemperingSampler):
            raise NotImplementedError(
                f"Sampler of type {type(sampler)} is not supported. Only MetropolisSampler and ParallelTemperingSampler are supported."
            )
        if sampler.n_chains % n_replicas != 0:
            raise ValueError(
                f"The number of replicas (n_replicas={n_replicas}) must divide the number of chains in the sampler (sampler.n_chains={sampler.n_chains})."
            )

        # Make a sampler that is sampler ⊗ FixedRule
        sampler = wrap_sampler(sampler, parameter_space, joint_space)

        super().__init__(hilbert=joint_space)
        self._hilbert_physical = hilbert_physical

        if model is not None:
            # extract init and apply functions
            # Wrap it in an HashablePartial because if two instances of the same model are provided,
            # model.apply and model2.apply will be different methods forcing recompilation, but
            # model and model2 will have the same hash.
            self._model_framework = model_frameworks.identify_framework(model)
            _maybe_unwrapped_variables, model = self._model_framework.wrap(model)

            if variables is None:
                if _maybe_unwrapped_variables is not None:
                    variables = _maybe_unwrapped_variables

            self._model = model

            self._init_fun = nkjax.HashablePartial(
                lambda model, *args, **kwargs: model.init(*args, **kwargs), model
            )
            self._apply_fun = wrap_to_support_scalar(
                nkjax.HashablePartial(
                    lambda model, pars, x, **kwargs: model.apply(pars, x, **kwargs),
                    model,
                )
            )
        else:
            raise NotImplementedError

        if variables is not None:
            self.variables = variables
        else:
            self.init(seed, dtype=sampler.dtype)

        if sampler_seed is None and seed is not None:
            key, key2 = jax.random.split(nkjax.PRNGKey(seed), 2)
            sampler_seed = key2

        self._sampler_seed = nkjax.PRNGKey(sampler_seed)
        self.sampler = sampler

        # default argument for n_samples/n_samples_per_rank
        if n_samples is None:
            # get the first multiple of sampler.n_chains above 1000 to avoid
            # printing a warning on construction
            self.n_samples = int(np.ceil(1000 / sampler.n_chains) * sampler.n_chains)
        elif n_samples is not None:
            self.n_samples = n_samples
        self.n_discard_per_chain = n_discard_per_chain  # type: ignore[assignment]

        if n_replicas is None:
            n_replicas = self.sampler.n_chains_per_rank
        self._n_replicas = n_replicas
        self._parameter_array = self.sampler_state.σ[
            :n_replicas, hilbert_physical.size :
        ].reshape(n_replicas, parameter_space.size)
        self.parameter_array = self._parameter_array

        self.chunk_size = chunk_size

    @property
    def hilbert_physical(self) -> AbstractHilbert:
        """
        The physical hilbert space of the variational state.
        """
        return self._hilbert_physical

    def init(self, seed=None, dtype=None):
        """
        Initialises the variational parameters of the variational state.
        """
        if self._init_fun is None:
            raise RuntimeError(
                "Cannot initialise the parameters of this state"
                "because you did not supply a valid init_function."
            )

        if dtype is None:
            dtype = self.sampler.dtype

        key = nkjax.PRNGKey(seed)

        dummy_input = self.hilbert.random_state(key, 1, dtype=dtype)

        if config.netket_experimental_sharding:
            par_sharding = NamedSharding(jax.sharding.get_abstract_mesh(), P())
        else:
            par_sharding = None
        variables = jax.jit(self._init_fun, out_shardings=par_sharding)(
            {"params": key}, dummy_input
        )
        self.variables = variables

    @property
    def _sampler_model(self) -> nn.Module:
        """Returns the model definition used for sampling this variational state.
        Equal to `.model`.
        """
        return self._model

    @property
    def _sampler_variables(self):
        """Returns the variables used for sampling this variational state.
        Equal to `.variables`
        """
        return self.variables

    @property
    def sampler(self) -> Sampler:
        """The Monte Carlo sampler used by this Monte Carlo variational state."""
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: Sampler):
        if not isinstance(sampler, Sampler):
            raise TypeError(
                f"The sampler should be a subtype of netket.sampler.Sampler, but {type(sampler)} is not."
            )

        self._sampler_seed, seed = jax.random.split(self._sampler_seed, 2)

        # Save the old `n_samples` before the new `sampler` is set.
        # `_chain_length == 0` means that this `MCState` is being constructed.
        if self._chain_length > 0:
            n_samples_old = self.n_samples

        self._sampler = sampler
        self.sampler_state = self.sampler.init_state(
            self._sampler_model, self._sampler_variables, seed=seed
        )
        self._sampler_state_previous = self.sampler_state

        # Update `n_samples`, `n_samples_per_rank`, and `chain_length` according
        # to the new `sampler.n_chains`.
        # If `n_samples` is divisible by the new `sampler.n_chains`, it will be
        # unchanged; otherwise it will be rounded up.
        # If the new `n_samples_per_rank` is not divisible by `chunk_size`, a
        # `ValueError` will be raised.
        # `_chain_length == 0` means that this `MCState` is being constructed.
        if self._chain_length > 0:
            self.n_samples = n_samples_old  # type: ignore

        self.reset()

    @property
    def n_samples(self) -> int:
        """The total number of samples generated at every sampling step."""
        return self.chain_length * self.sampler.n_chains

    @n_samples.setter
    def n_samples(self, n_samples: int):
        chain_length = compute_chain_length(self.sampler.n_chains, n_samples)
        self.chain_length = chain_length

    @property
    def n_samples_per_rank(self) -> int:
        """The number of samples generated on every jax device at every sampling step."""
        return self.chain_length * self.sampler.n_chains_per_rank

    @n_samples_per_rank.setter
    def n_samples_per_rank(self, n_samples_per_rank: int):
        self.n_samples = n_samples_per_rank * sharding.device_count()

    @property
    def chain_length(self) -> int:
        """
        Length of the markov chain used for sampling configurations.
        """
        return self._chain_length

    @chain_length.setter
    def chain_length(self, chain_length: int):
        if chain_length <= 0:
            raise ValueError(f"Invalid chain length: chain_length={chain_length}")

        self._chain_length = chain_length
        self.reset()

    @property
    def n_discard_per_chain(self) -> int:
        """
        Number of discarded samples at the beginning of the markov chain.
        """
        return self._n_discard_per_chain

    @n_discard_per_chain.setter
    def n_discard_per_chain(self, n_discard_per_chain: int | None):
        if n_discard_per_chain is not None and n_discard_per_chain < 0:
            raise ValueError(
                f"Invalid number of discarded samples: n_discard_per_chain={n_discard_per_chain}"
            )

        # don't discard if the sampler is exact
        if self.sampler.is_exact:
            if n_discard_per_chain is not None and n_discard_per_chain > 0:
                warnings.warn(
                    "An exact sampler does not need to discard samples. Setting n_discard_per_chain to 0.",
                    stacklevel=2,
                )
            n_discard_per_chain = 0

        self._n_discard_per_chain = (
            int(n_discard_per_chain) if n_discard_per_chain is not None else 5
        )

    @property
    def chunk_size(self) -> int | None:
        """
        Suggested *maximum size* of the chunks used in forward and backward evaluations
        of the Neural Network model.

        If your inputs are smaller than the chunk size this setting is ignored.

        This can be used to lower the memory required to run a computation with a very
        high number of samples or on a very large lattice. Notice that inputs and
        outputs must still fit in memory, but the intermediate computations will now
        require less memory.

        This option comes at an increased computational cost. While this cost should
        be negligible for large-enough chunk sizes, don't use it unless you are memory
        bound!

        This option is an hint: only some operations support chunking. If you perform
        an operation that is not implemented with chunking support, it will fall back
        to no chunking. To check if this happened, set the environment variable
        `NETKET_DEBUG=1`.
        """
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, chunk_size: int | None):
        # disable chunks if it is None
        if chunk_size is None:
            self._chunk_size = None
            return

        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(
                f"Chunk size must be a positive INTEGER (got {chunk_size} instead)."
            )

        if not _is_power_of_two(chunk_size):
            warnings.warn(
                "For performance reasons, we suggest to use a power-of-two chunk size.",
                stacklevel=2,
            )

        self._chunk_size = chunk_size

    def reset(self):
        """
        Resets the sampled states. This method is called automatically every time
        that the parameters/state is updated.
        """
        self._samples = None

    @timing.timed
    def sample(
        self,
        *,
        chain_length: int | None = None,
        n_samples: int | None = None,
        n_discard_per_chain: int | None = None,
    ) -> jnp.ndarray:
        """
        Sample a certain number of configurations.

        If one among chain_length or n_samples is defined, that number of samples
        are generated. Otherwise the value set internally is used.

        Args:
            chain_length: The length of the markov chains.
            n_samples: The total number of samples across all devices.
            n_discard_per_chain: Number of discarded samples at the beginning of the markov chain.
        """

        if n_samples is None and chain_length is None:
            chain_length = self.chain_length
        else:
            if chain_length is not None and n_samples is not None:
                raise ValueError("Cannot specify both `chain_length` and `n_samples`.")
            elif chain_length is None:
                chain_length = compute_chain_length(self.sampler.n_chains, n_samples)

        if n_discard_per_chain is None:
            n_discard_per_chain = self.n_discard_per_chain

        # Store the previous sampler state, for serialization purposes
        self._sampler_state_previous = self.sampler_state

        self.sampler_state = self.sampler.reset(
            self._sampler_model, self._sampler_variables, self.sampler_state
        )

        if self.n_discard_per_chain > 0:
            with timing.timed_scope("sampling n_discarded samples") as timer:
                _, self.sampler_state = self.sampler.sample(
                    self._sampler_model,
                    self.variables,
                    state=self.sampler_state,
                    chain_length=n_discard_per_chain,
                )
                # This won't actually block unless we are really timing
                timer.block_until_ready(_)

        self._samples, self.sampler_state = self.sampler.sample(
            self._sampler_model,
            self._sampler_variables,
            state=self.sampler_state,
            chain_length=chain_length,
        )
        return self._samples

    @property
    def samples(self) -> jax.Array:
        """
        Returns the set of cached samples.

        The samples returned are guaranteed valid for the current state of
        the variational state. If no cached parameters are available, then
        they are sampled first and then cached.

        To obtain a new set of samples either use
        :meth:`~MCState.reset` or :meth:`~MCState.sample`.
        """
        if self._samples is None:
            self.sample()
        return self._samples  # type: ignore[return-value]

    def log_value(self, σ: jnp.ndarray) -> jnp.ndarray:
        r"""
        Evaluate the variational state for a batch of states and returns
        the logarithm of the amplitude of the quantum state.

        For pure states,
        this is :math:`\log(\langle\sigma|\psi\rangle)`, whereas for mixed states
        this is :math:`\log(\langle\sigma_r|\rho|\sigma_c\rangle)`, where
        :math:`\psi` and :math:`\rho` are respectively a pure state
        (wavefunction) and a mixed state (density matrix).
        For the density matrix, the left and right-acting states (row and column)
        are obtained as :code:`σr=σ[::,0:N]` and :code:`σc=σ[::,N:]`.

        Given a batch of inputs :code:`(Nb, N)`, returns a batch of outputs
        :code:`(Nb,)`.
        """
        return jit_evaluate(self._apply_fun, self.variables, σ)

    @timing.timed
    def local_estimators(self, op, *, chunk_size: int | None = None) -> LocalEstimators:
        if chunk_size is None:
            chunk_size = self.chunk_size

        return nk_local_estimators(self, op, chunk_size)

    @property
    def parameter_array(self) -> jax.Array:
        return self._parameter_array
        # x = self.sampler_state.σ
        # return x[..., self.hilbert_physical.size :]

    @parameter_array.setter
    def parameter_array(self, parameter_array: jax.Array):
        assert parameter_array.ndim == 2
        assert parameter_array.shape[0] == self.n_replicas
        assert parameter_array.shape[-1] == self.parameter_array.shape[-1]

        self._parameter_array = parameter_array

        # new_ss = self.sampler_state.replace(
        #    σ=σ.at[..., self.hilbert_physical.size :].set(parameter_array)
        # )
        self.sampler_state = self.sampler_state.replace(
            σ=replace_parameters(self.sampler_state.σ, parameter_array)
        )
        self.reset()

    @property
    def n_replicas(self) -> int:
        return self._n_replicas

    def get_state(self, parameters: jax.Array, seed=None) -> MCState:
        """
        Given a set of parameters, returns a standard MCState instance that corresponds
        to the foundational state with those parameters.

        Args:
            parameters: The parameters to be used for the foundational state.
        """
        assert parameters.ndim == 1
        assert parameters.shape[0] == self.parameter_space.size

        model_instance = FoundationalInstance(self.parameter_space, self._model)

        variables = {
            "foundational": {"parameters": jnp.asarray(parameters)},
            "params": self.parameters,
        }

        if seed is None:
            self._sampler_seed, seed = jax.random.split(self._sampler_seed, 2)
        else:
            seed = jax.random.PRNGKey(seed)

        vstate = MCState(
            self._physical_sampler,
            model_instance,
            n_samples=self.n_samples,
            n_discard_per_chain=self.n_discard_per_chain,
            chunk_size=self.chunk_size,
            variables=variables,
            seed=seed,
        )
        # Use current samples
        vstate.sampler_state = vstate.sampler_state.replace(
            σ=self.sampler_state.σ[..., : self.hilbert_physical.size].astype(
                vstate.sampler_state.σ.dtype
            )
        )
        return vstate

    def replace_sampler_seed(self, seed: SeedT | None = None):
        """Change the rng state of the sampler contained in this Monte Carlo state.

        The use-case for this is when you create a copy of a state (for example by
        loading it from disk), but you want the copy to generate samples that are
        not correlated to the ones of the original state.

        Beware that for this to work correctly, you probably need to resample a bunch
        of times in order to decorrelate the chains, because this method only changes
        the rng seed, but not the current configurations in the chain.

        Args:
            seed: The seed to use. If None, a random seed is drawn.
        """
        seed = nkjax.PRNGKey(seed)
        self._sampler_seed = seed
        self.sampler_state = self.sampler_state.replace(rng=seed)
        self.reset()

    def save(self, path):  # noqa: F811
        _save({"state": self}, path)

    @classmethod
    def load(cls, path, new_seed: bool | int = True):
        vstate = _load(path)["state"]
        if new_seed is not False:
            if new_seed is True:
                new_seed = None
            vstate.replace_sampler_seed(new_seed)
        return vstate

    def is_state(
        self,
        target_parameters: jax.Array,
        *,
        reference=None,
        chunk_size: int | None = None,
    ):
        """
        Create an :class:`~netket_foundation.expectation_value.ISState`
        targeting ``target_parameters``.

        By default the current physical samples are used as the IS reference,
        with log-probabilities computed from the joint samples (which carry
        each replica's foundational parameters), so IS weights correctly
        account for the per-replica distribution.

        Alternatively, pass ``reference`` to use a precomputed reference
        distribution — either a path to a ``.npz`` file written by
        :meth:`~netket_foundation.expectation_value.SamplesWithProb.save`, a
        :class:`~netket_foundation.expectation_value.SamplesWithProb`, or a raw
        ``(samples, log_probs)`` tuple.

        Args:
            target_parameters: 1-D array of foundational parameters for the
                               target state.
            reference:         Optional IS reference: a ``.npz`` path, a
                               :class:`SamplesWithProb`, or a
                               ``(samples, log_probs)`` tuple. Defaults to the
                               current physical samples.
            chunk_size:        Forwarded to ISState; defaults to self.chunk_size.

        Returns:
            ISState ready for .expect().
        """
        from netket_foundation._src.expectation_value.is_state import ISState
        from netket_foundation._src.expectation_value.samples_with_probability import (
            SamplesWithProb,
        )
        from netket_foundation._src.vqs.io import samples_with_probability

        # Resolve the reference into (physical samples, log_probs).
        if reference is None:
            # Default: the current physical samples. samples_with_probability
            # strips the parameter columns and evaluates log-probs on the joint
            # samples (so each row uses its own replica parameters).
            reference = samples_with_probability(self)
        elif isinstance(reference, str | os.PathLike):
            reference = SamplesWithProb.load(reference)

        if isinstance(reference, SamplesWithProb):
            samples = reference.samples
            log_probs_ref = reference.log_probs
        else:
            # Raw (samples, log_probs) tuple.
            samples, log_probs_ref = reference

        # Target apply_fn: FoundationalInstance (physical) model at target_parameters
        model_instance = FoundationalInstance(self.parameter_space, self._model)
        target_apply_fn = wrap_to_support_scalar(
            nkjax.HashablePartial(lambda m, p, x: m.apply(p, x), model_instance)
        )
        target_vars = {
            "foundational": {"parameters": jnp.asarray(target_parameters)},
            "params": self.parameters,
        }

        return ISState(
            samples,
            log_probs_ref,
            target_apply_fn,
            target_vars,
            chunk_size=chunk_size or self.chunk_size,
        )


@jax.jit
def replace_parameters(samples, parameters):
    Ns, NM = samples.shape
    Np, M = parameters.shape
    # N = NM - M

    # Reshape samples to (Ns//Np, Np, N+M)
    samples_reshaped = samples.reshape(Np, Ns // Np, NM)

    # Broadcast parameters to shape (Ns//Np, Np, M)
    param_broadcast = jnp.broadcast_to(parameters.reshape(Np, 1, M), (Np, Ns // Np, M))

    # Use .at[] to update the last M entries of the last axis
    samples_updated = samples_reshaped.at[..., -M:].set(param_broadcast)

    # Optionally reshape back to (Ns, N+M)
    samples_final = samples_updated.reshape(Ns, NM)
    return jax.lax.with_sharding_constraint(
        samples_final, NamedSharding(jax.sharding.get_abstract_mesh(), P("S"))
    )


@dispatch.dispatch
def get_local_kernel_arguments(  # noqa: F811
    vstate: FoundationalQuantumState, Ô: ParametrizedOperator
):
    check_hilbert(vstate.hilbert, Ô.hilbert)
    σ = vstate.samples
    return σ, Ô


@dispatch.dispatch
def get_local_kernel(  # noqa: F811
    vstate: FoundationalQuantumState, Ô: ParametrizedOperator
):
    return foundational_kernel_jax


@dispatch.dispatch
def get_local_kernel(  # noqa: F811
    vstate: FoundationalQuantumState, Ô: ParametrizedOperator, chunk_size: int
):
    return foundational_kernel_jax_chunked


def foundational_kernel_jax(
    logpsi: Callable, pars: PyTree, σ: jax.Array, O: ParametrizedOperator
):
    """
    local_value kernel for MCState for jax-compatible operators
    """
    σp, mel = O.get_conn_padded(σ)
    logpsi_σ = logpsi(pars, σ)
    logpsi_σp = logpsi(pars, σp.reshape(-1, σp.shape[-1])).reshape(σp.shape[:-1])
    return jnp.sum(mel * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1)


def foundational_kernel_jax_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: jax.Array,
    O: ParametrizedOperator,
    *,
    chunk_size: int | None = None,
):
    """
    Chunked version of `foundational_kernel_jax`, mirroring
    netket.vqs.mc.mc_state.kernels.local_value_kernel_jax_chunked.
    """
    σp, mel = O.get_conn_padded(σ)
    N = σ.shape[-1]

    logpsi_chunked = nkjax.vmap_chunked(
        partial(logpsi, pars), in_axes=0, chunk_size=chunk_size
    )

    logpsi_σ = logpsi_chunked(σ.reshape((-1, N))).reshape(σ.shape[:-1])  # (batch,)
    logpsi_σp = logpsi_chunked(σp.reshape((-1, N))).reshape(
        σp.shape[:-1]
    )  # (batch, n_conn)

    # Same formula as your non-chunked kernel, but with the chunked evaluations.
    return jnp.sum(mel * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1)


@nk_local_estimators.dispatch
def _foundational_local_estimators(
    vstate: FoundationalQuantumState,
    op: ParametrizedOperator,
    chunk_size: int | None,
) -> LocalEstimators:
    σ, extra_args = get_local_kernel_arguments(vstate, op)
    shape = σ.shape
    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, shape[-1]))

    if chunk_size is None:
        data = jax.jit(foundational_kernel_jax, static_argnames=("logpsi",))(
            vstate._apply_fun, vstate.variables, σ, extra_args
        )
    else:
        data = jax.jit(
            foundational_kernel_jax_chunked,
            static_argnames=("logpsi", "chunk_size"),
        )(
            vstate._apply_fun,
            vstate.variables,
            σ,
            extra_args,
            chunk_size=chunk_size,
        )

    return LocalEstimators(data.reshape(shape[:-1]))


@dispatch.dispatch
def expect(  # noqa: F811
    vstate: FoundationalQuantumState,
    Ô: ParametrizedOperator,
    chunk_size: int | None = None,
):
    σ, args = get_local_kernel_arguments(vstate, Ô)
    local_estimator_fun = get_local_kernel(vstate, Ô)

    return _expect(
        local_estimator_fun,
        vstate._apply_fun,
        vstate.sampler.machine_pow,
        vstate.parameters,
        vstate.model_state,
        σ,
        args,
        n_replicas=vstate.n_replicas,
    )


@partial(jax.jit, static_argnums=(0, 1, 7))
def _expect(
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    machine_pow: int,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
    n_replicas: int,
) -> Stats:
    n_chains = σ.shape[0]
    if σ.ndim >= 3:
        σ = jax.lax.collapse(σ, 0, 2)

    def logpsi(w, σ):
        return model_apply_fun({"params": w, **model_state}, σ)

    def log_pdf(w, σ):
        return machine_pow * model_apply_fun({"params": w, **model_state}, σ).real

    L_σ = local_value_kernel(logpsi, parameters, σ, local_value_args)
    L_σ = L_σ.reshape(n_replicas, n_chains // n_replicas, -1)
    Ō_stats = [statistics(L_σ[i]) for i in range(n_replicas)]

    return Ō_stats


from netket_foundation._src.vqs import serialize as serialize  # noqa: E402
from netket_foundation._src.vqs import (  # noqa: E402  (registers SamplesWithProb)
    io as io,
)
