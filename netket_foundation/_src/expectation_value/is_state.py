"""
ISState: importance-sampling view of a target variational state.

The reference distribution is supplied as raw (samples, log_probs_ref) —
decoupled from any MCState so that data can come from files, different codes,
or a sweep of candidate references.

Weights are computed lazily and cached; all .expect() calls share them.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from netket import jax as nkjax
from netket.vqs import MCState, expect as nk_expect
from netket.utils.types import PyTree
from netket.operator._abstract_operator import AbstractOperator

from netket_foundation._src.expectation_value.importance_sampling import (
    ISResult,
    ISWeights,
    _is_weights,
)
from netket_foundation._src.operator.parametrized import ParametrizedOperator
from netket_foundation._src.vqs.io import samples_with_probability


class ISState:
    """
    Importance-sampled view: physical ``samples`` drawn from a reference
    distribution (characterized by ``log_probs_ref``) evaluated at
    ``variables`` under ``apply_fn``.

    IS weights are computed once and cached across successive .expect() calls.

    Typical construction:

        # From a live MCState reference, reweighting to target parameters
        is_st = ISState.from_mc_state(mc_ref, target_params)

        # From a self-contained nqxpack file (state + reference bundled)
        is_st = ISState.from_nqxpack("ref.nk", target_params)

        # From lightweight .npz arrays paired with a live target state
        is_st = vs.is_state(pars, reference=SamplesWithProb.load("ref.npz"))

        # From raw arrays directly
        is_st = ISState(samples, log_probs, apply_fn, variables)
    """

    def __init__(
        self,
        samples: jax.Array,
        log_probs_ref: jax.Array,
        apply_fn: Callable,
        variables: PyTree,
        *,
        chunk_size: int | None = None,
    ):
        """
        Args:
            samples:          Physical configurations, shape (n_samples, N).
            log_probs_ref:    Reference log-density ``log p_ref(sigma)`` at each
                              sample, shape (n_samples,).
            apply_fn:         Model apply function for the *target* state;
                              signature (variables, x) -> log_amplitude.
            variables:        Variables of the (target) state being evaluated.
            chunk_size:       Optional chunk size for model evaluation.
        """
        if samples.ndim != 2:
            raise ValueError(
                "`samples` must be 2D (n_samples, N) — for a "
                "FoundationalQuantumState the parameter columns must already be "
                f"stripped. Got shape {samples.shape}."
            )
        if log_probs_ref.ndim != 1:
            raise ValueError(
                f"`log_probs_ref` must be 1D (n_samples,), got shape "
                f"{log_probs_ref.shape}."
            )
        if samples.shape[0] != log_probs_ref.shape[0]:
            raise ValueError(
                f"`samples` and `log_probs_ref` disagree on n_samples: "
                f"{samples.shape[0]} vs {log_probs_ref.shape[0]}."
            )

        self._samples = samples
        self._log_probs_ref = log_probs_ref
        self._apply_fn = apply_fn
        self._variables = variables
        self._chunk_size = chunk_size
        self._cache: ISWeights | None = None  # lazy

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_mc_state(
        cls,
        mc_ref: MCState,
        target: PyTree,
        *,
        chunk_size: int | None = None,
    ) -> ISState:
        """
        Build an ISState from an already-sampled MCState reference.

        ``mc_ref`` supplies both the reference distribution (its samples and
        log-probabilities) and the model; ``target`` selects the parameters that
        model is evaluated at for the target distribution.

        Args:
            mc_ref: Sampled reference state. Provides the samples *and* the
                    target model (``mc_ref._apply_fun``).
            target: Target foundational parameters (a 1-D array). The target
                    variables reuse ``mc_ref``'s variable tree with only the
                    ``"foundational"`` parameters swapped. A full variables dict
                    may be passed instead for a plain MCState reference that has
                    no foundational collection.
        """
        reference = samples_with_probability(mc_ref)
        # Mapping: full variables dict as-is; array: graft params onto mc_ref's tree.
        variables = (
            target
            if isinstance(target, Mapping)
            else {
                **mc_ref.variables,
                "foundational": {"parameters": jnp.asarray(target)},
            }
        )
        return cls(
            reference.samples,
            reference.log_probs,
            mc_ref._apply_fun,
            variables,
            chunk_size=chunk_size,
        )

    @classmethod
    def from_nqxpack(
        cls,
        path: str,
        target: PyTree,
        *,
        key: str = "state",
        chunk_size: int | None = None,
    ) -> ISState:
        """
        Load a state saved by :func:`netket_foundation.vqs.save` and use it as
        the IS reference, reweighting to the target parameters ``target``.

        If the file bundles a reference distribution (``SamplesWithProb``), its
        samples and log-probabilities are used directly. Otherwise the reference
        is reconstructed from the loaded state's samples (drawing them if the
        state has not been sampled yet).

        Args:
            path:   Path to the .nk file written by ``nkf.vqs.save`` / nqxpack.
            target: Target foundational parameters (a 1-D array), or a full
                    variables dict for a plain MCState reference.
            key:    Key under which the state was stored (default "state").
        """
        # Local imports: break the is_state <-> state import cycle (only needed
        # at call time, not at module load).
        from netket_foundation._src.vqs.io import load as load_state
        from netket_foundation._src.vqs.state import FoundationalQuantumState

        data = load_state(path, return_dict=True, key=key)
        state = data[key]
        reference = data.get("reference")  # SamplesWithProb, or None if not bundled

        if isinstance(state, FoundationalQuantumState):
            params = (
                target["foundational"]["parameters"]
                if isinstance(target, Mapping)
                else target
            )
            return state.is_state(params, reference=reference, chunk_size=chunk_size)

        # Plain MCState: _apply_fun and samples are both physical.
        if reference is None:
            reference = samples_with_probability(state)
        # Mapping: full variables dict as-is; array: graft params onto state's tree.
        variables = (
            target
            if isinstance(target, Mapping)
            else {
                **state.variables,
                "foundational": {"parameters": jnp.asarray(target)},
            }
        )
        return cls(
            reference.samples,
            reference.log_probs,
            state._apply_fun,
            variables,
            chunk_size=chunk_size,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def samples(self) -> jax.Array:
        return self._samples

    @property
    def variables(self) -> PyTree:
        return self._variables

    @property
    def apply_fn(self) -> Callable:
        return self._apply_fn

    @property
    def n_samples(self) -> int:
        return int(self._samples.shape[0])

    # ------------------------------------------------------------------
    # IS weights (cached)
    # ------------------------------------------------------------------

    def _compute_weights(self) -> ISWeights:
        if self._cache is None:
            # TODO: This should not be fixed at pow=2, but for now it's good enough..
            log_prob_target = 2.0 * self._apply_fn(self._variables, self._samples).real
            self._cache = _is_weights(log_prob_target, self._log_probs_ref)
        return self._cache

    @property
    def normalized_weights(self) -> jax.Array:
        """IS weights normalized to sum 1."""
        return self._compute_weights().normalized_weights

    @property
    def ess(self) -> float:
        return self._compute_weights().ess

    @property
    def ess_fraction(self) -> float:
        return self.ess / self.n_samples

    # ------------------------------------------------------------------
    # Estimator — delegates to nk.vqs.expect dispatch
    # ------------------------------------------------------------------

    def expect(self, observable, *, chunk_size: int | None = None):
        """
        Compute the IS-weighted expectation value of an observable.

        Dispatches on the observable type via ``nk.vqs.expect``.  The default
        overload handles any ``AbstractOperator`` via local values.  Custom
        observables (e.g. ``SusceptibilityObservable``) register their own
        overload with ``@nk_expect.dispatch``.

        IS weights are cached after the first call and reused across calls.
        """
        return nk_expect(self, observable, chunk_size or self._chunk_size)

    def __repr__(self):
        return (
            f"ISState(n_samples={self.n_samples}"
            + (f", ess={self.ess:.1f}" if self._cache is not None else "")
            + ")"
        )


# ---------------------------------------------------------------------------
# Operator overloads
# ---------------------------------------------------------------------------


@nk_expect.dispatch
def expect_isstate_parametrized_operator(
    is_state: ISState,
    observable: ParametrizedOperator,
    chunk_size: int | None,
) -> ISResult:
    params = is_state.variables["foundational"]["parameters"]
    if params.ndim != 1:
        raise ValueError(
            "ISState expects a single target parameter vector for "
            f"ParametrizedOperator evaluation, got shape {params.shape}."
        )

    return expect_isstate_operator(is_state, observable.with_params(params), chunk_size)


@nk_expect.dispatch
def expect_isstate_operator(
    is_state: ISState,
    observable: AbstractOperator,
    chunk_size: int | None,
) -> ISResult:
    O_loc = _local_values(
        is_state._apply_fn,
        is_state._variables,
        is_state._samples,
        observable,
        chunk_size,
    )
    w = is_state.normalized_weights
    mean = jnp.sum(w * O_loc)
    # `variance` is the variance of O under the target (weighted population
    # variance); `error_of_mean` is the standard error of the self-normalized
    # IS estimator via the delta method, sqrt(sum_i w_i^2 (O_i - mean)^2). This
    # is exact (asymptotically) for arbitrary weights and reduces to the plain
    # MC error sqrt(var/n) when the weights are uniform (w_i = 1/n).
    sq_dev = jnp.abs(O_loc - mean) ** 2
    variance = jnp.real(jnp.sum(w * sq_dev))
    error_of_mean = jnp.sqrt(jnp.real(jnp.sum(w**2 * sq_dev)))

    return ISResult(
        mean=mean,
        error_of_mean=error_of_mean,
        variance=variance,
        ess=is_state.ess,
        ess_fraction=is_state.ess_fraction,
        n_samples=is_state.n_samples,
    )


# ---------------------------------------------------------------------------
# Private helper: local operator values
# ---------------------------------------------------------------------------


def _local_values(apply_fn, variables, samples, operator, chunk_size):
    # get_conn_padded runs on the host for non-jax operators, so it stays
    # outside the jitted kernel.
    sigma_p, mels = operator.get_conn_padded(samples)
    return _local_values_kernel(apply_fn, variables, samples, sigma_p, mels, chunk_size)


@partial(jax.jit, static_argnames=("apply_fn", "chunk_size"))
def _local_values_kernel(apply_fn, variables, samples, sigma_p, mels, chunk_size):
    n_samples, N = samples.shape[0], samples.shape[-1]
    n_conn = sigma_p.shape[1]

    _apply = nkjax.apply_chunked(
        lambda s: apply_fn(variables, s), in_axes=0, chunk_size=chunk_size
    )

    log_psi_sigma = _apply(samples)
    log_psi_conn = _apply(sigma_p.reshape(-1, N)).reshape(n_samples, n_conn)

    return jnp.sum(mels * jnp.exp(log_psi_conn - log_psi_sigma[:, None]), axis=-1)
