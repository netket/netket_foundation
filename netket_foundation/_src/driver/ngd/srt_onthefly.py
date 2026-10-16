from collections.abc import Callable
from functools import partial

from einops import rearrange

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from jax.sharding import NamedSharding, PartitionSpec as P

from netket import jax as nkjax
from netket import config
from netket.jax._jacobian.default_mode import JacobianMode
from netket.utils import timing
from netket.utils.types import Array

from netket.jax import _ntk as nt


@timing.timed
@partial(
    jax.jit,
    static_argnames=(
        "log_psi",
        "solver_fn",
        "n_replicas",
        "chunk_size",
        "mode",
    ),
)
def srt_onthefly(
    log_psi,
    local_energies,
    parameters,
    model_state,
    samples,
    *,
    n_replicas: int,
    diag_shift: float | Array,
    solver_fn: Callable[[Array, Array], Array],
    mode: JacobianMode,
    proj_reg: float | Array | None = None,
    momentum: float | Array | None = None,
    old_updates: Array | None = None,
    chunk_size: int | None = None,
    pdf: Array | None = None,
):
    r"""
    On-the-fly NTK (minSR/kernel-SR) update for the foundation training scheme.

    Computes the same natural-gradient update as
    :func:`netket_foundation._src.driver.ngd.sr_srt_common.srt` (``use_ntk=True``)
    but without ever materialising the dense ``(N_mc, n_params)`` Jacobian: the
    NTK gram matrix is built with :func:`netket.jax.empirical_ntk_by_jacobian`
    and the back-projection to parameter space is done with a chunked VJP.

    The only difference w.r.t. :func:`netket._src.ngd.srt_onthefly.srt_onthefly`
    is that every centering (the RHS, the momentum term, the NTK and the
    auxiliary vector) is performed **per replica** instead of globally: the
    foundation state holds ``n_replicas`` replicas blocked replica-major along
    the sample axis, all sharing the same parameters, and each replica must be
    centered against its own mean. Concretely, the sample axis is kept as
    ``(n_replicas, ns)`` (with ``ns = N_mc // n_replicas``) and every centering
    is a single mean over the sample-within-replica axis ``ns``; arrays are only
    flattened back to the bare sample axis at the autodiff/solver boundary
    (NTK contraction, linear solve, VJP). The NTK gram itself is the full
    ``N_mc x N_mc`` matrix mixing all replicas, exactly as in the dense path.
    """
    if pdf is not None:
        raise NotImplementedError(
            "Importance-sampling weights (pdf) are not yet supported in "
            "srt_onthefly. Use on_the_fly=False for FullSum/importance-sampling "
            "states."
        )

    N_mc = local_energies.size
    ns = N_mc // n_replicas  # samples per replica

    # Split all parameters into real and imaginary parts separately
    parameters_real, rss = nkjax.tree_to_real(parameters)

    # complex: (Nmc) -> (Nmc,2) - splitting real and imaginary output like 2 classes
    # real:    (Nmc) -> (Nmc,)  - no splitting
    def _apply_fn(parameters_real, samples, model_state):
        variables = {"params": rss(parameters_real), **model_state}
        log_amp = log_psi(variables, samples)

        if mode == "complex":
            re, im = log_amp.real, log_amp.imag
            return jnp.concatenate(
                (re[:, None], im[:, None]), axis=-1
            )  # shape [N_mc,2]
        else:
            return log_amp.real  # shape [N_mc, ]

    def jvp_f_chunk(parameters, model_state, vector, samples):
        r"""
        Creates the jvp of the function `_apply_fn` with respect to the parameters.
        This jvp is then evaluated in chunks of `chunk_size` samples.
        """
        f = lambda params: _apply_fn(params, samples, model_state=model_state)
        _, acc = jax.jvp(f, (parameters,), (vector,))
        return acc

    # Compute the rhs of the linear system, kept in (n_replicas, ns, ...) shape
    # and centered with a single mean over the sample-within-replica axis.
    de = local_energies.reshape(n_replicas, ns)  # (n_replicas, ns)
    de = de - de.mean(axis=1, keepdims=True)

    # At the moment the final vjp is centered by centering the auxiliary vector a.
    # This is the same as centering the jacobian but may have larger variance.
    dv = 2.0 * de / jnp.sqrt(N_mc)  # (n_replicas, ns)
    if mode == "complex":
        dv = jnp.stack([dv.real, dv.imag], axis=-1)  # (n_replicas, ns, 2)
    else:
        dv = dv.real  # (n_replicas, ns)

    if momentum is not None:
        if old_updates is None:
            old_updates = tree_map(jnp.zeros_like, parameters_real)
        else:
            acc = nkjax.apply_chunked(
                jvp_f_chunk, in_axes=(None, None, None, 0), chunk_size=chunk_size
            )(parameters_real, model_state, old_updates, samples)

            # center per replica (same operation as centering the jacobian rows)
            acc = acc.reshape((n_replicas, ns) + acc.shape[1:])
            acc = (acc - acc.mean(axis=1, keepdims=True)) / jnp.sqrt(N_mc)
            dv -= momentum * acc

    # Flatten to the bare sample axis expected by the solver:
    # (n_replicas, ns[, 2]) -> [N_mc,] (real) or [2*N_mc,] (complex, sample-major then re/im)
    dv = dv.reshape(-1)

    # Collect all samples on all MPI ranks, those label the columns of the T matrix
    all_samples = samples
    if config.netket_sharding:
        samples = jax.lax.with_sharding_constraint(
            samples, NamedSharding(jax.sharding.get_abstract_mesh(), P("S", None))
        )
        all_samples = jax.lax.with_sharding_constraint(
            samples, NamedSharding(jax.sharding.get_abstract_mesh(), P())
        )

    _jacobian_contraction = nt.empirical_ntk_by_jacobian(
        f=_apply_fn,
        trace_axes=(),
        vmap_axes=0,
    )

    def jacobian_contraction(samples, all_samples, parameters_real, model_state):
        if config.netket_sharding:
            parameters_real = nkjax.lax.pcast(parameters_real, "S", to="varying")
        if chunk_size is None:
            # STRUCTURED_DERIVATIVES returns a complex array, but the imaginary part is zero
            # shape [N_mc/p.size, N_mc, 2, 2]
            return _jacobian_contraction(
                samples, all_samples, parameters_real, model_state=model_state
            ).real
        else:
            _all_samples, _ = nkjax.chunk(all_samples, chunk_size=chunk_size)
            ntk_local = jax.lax.map(
                lambda batch_lattice: (
                    _jacobian_contraction(
                        samples, batch_lattice, parameters_real, model_state=model_state
                    ).real
                ),
                _all_samples,
            )
            if mode == "complex":
                return rearrange(ntk_local, "nbatches i j z w -> i (nbatches j) z w")
            else:
                return rearrange(ntk_local, "nbatches i j -> i (nbatches j)")

    # If we are sharding, use shard_map manually
    if config.netket_sharding:
        mesh = jax.sharding.get_abstract_mesh()
        # SAMPLES, ALL_SAMPLES PARAMETERS_REAL
        in_specs = (P("S", None), P(), P(), P())
        out_specs = P("S", None)

        # By default, I'm not sure whether the jacobian_contraction of NeuralTangents
        # Is correctly automatically sharded across devices. So we force it to be
        # sharded with shard map to be sure

        jacobian_contraction = jax.shard_map(
            jacobian_contraction,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
        )

    # This disables the nkjax.sharding_decorator in here, which might appear
    # in the apply function inside.
    with nkjax.sharding._increase_SHARD_MAP_STACK_LEVEL():
        ntk_local = jacobian_contraction(
            samples, all_samples, parameters_real, model_state
        ).real

    # shape [N_mc, N_mc, 2, 2] or [N_mc, N_mc]
    if config.netket_sharding:
        # this sharding constraint should be useless, but let's keep it for safety.
        ntk = jax.lax.with_sharding_constraint(
            ntk_local, NamedSharding(jax.sharding.get_abstract_mesh(), P("S", None))
        )
    else:
        ntk = ntk_local
    if mode == "complex":
        # shape [2*N_mc, 2*N_mc] checked with direct calculation of J^T J
        ntk = rearrange(ntk, "i j z w -> (i z) (j w)")

    # Center the NTK per replica, avoiding the construction of a big dense
    # projector to lower memory pressure. This computes (1 / N_mc) * P K P,
    # where P is the block-diagonal projector that subtracts, within each
    # replica block, the mean over that replica's samples. Equivalent to
    # centering the (per-replica) jacobian rows in the dense path.
    #
    # This is the exact same additive centering as netket's global srt_onthefly
    # (`ntk - mean_rows - mean_cols + mean_both`); the only difference is that
    # the sample axis is split as `(n_replicas, ns)` and the means are taken
    # over the sample-within-replica axis `ns` rather than the full axis.
    if mode == "complex":
        # (a, p, z, b, q, w): a,b = replica, p,q = sample-in-replica, z,w = re/im
        ntk = ntk.reshape(n_replicas, ns, 2, n_replicas, ns, 2)
        mean_rows = ntk.mean(axis=1, keepdims=True)  # over row samples p
        mean_cols = ntk.mean(axis=4, keepdims=True)  # over column samples q
        mean_both = ntk.mean(axis=(1, 4), keepdims=True)
        ntk = ntk - mean_rows - mean_cols + mean_both
        ntk = ntk.reshape(2 * N_mc, 2 * N_mc)
    else:
        # (a, p, b, q): a,b = replica, p,q = sample-in-replica
        ntk = ntk.reshape(n_replicas, ns, n_replicas, ns)
        mean_rows = ntk.mean(axis=1, keepdims=True)  # over row samples p
        mean_cols = ntk.mean(axis=3, keepdims=True)  # over column samples q
        mean_both = ntk.mean(axis=(1, 3), keepdims=True)
        ntk = ntk - mean_rows - mean_cols + mean_both
        ntk = ntk.reshape(N_mc, N_mc)

    ntk = ntk / N_mc

    # Create identity matrix with same sharding as ntk: P("S", None)
    if config.netket_sharding:
        local_size = ntk.shape[0]
        identity = jnp.eye(local_size, dtype=ntk.dtype)
        identity = jax.lax.with_sharding_constraint(
            identity, NamedSharding(jax.sharding.get_abstract_mesh(), P("S", None))
        )
    else:
        identity = jnp.eye(ntk.shape[0], dtype=ntk.dtype)

    # add diag shift
    ntk_shifted = ntk + diag_shift * identity

    # add projection regularization
    if proj_reg is not None:
        ntk_shifted = ntk_shifted + proj_reg / N_mc

    # some solvers return a tuple, some others do not.
    aus_vector = solver_fn(ntk_shifted, dv)

    if isinstance(aus_vector, tuple):
        aus_vector, info = aus_vector
        if info is None:
            info = {}
    else:
        info = {}

    if config.netket_sharding:
        aus_vector = jax.lax.with_sharding_constraint(
            aus_vector,
            NamedSharding(jax.sharding.get_abstract_mesh(), P("S")),
        )

    aus_vector = jnp.squeeze(aus_vector)

    # Center the vector per replica, equivalent to centering the jacobian rows
    # in the back-projection. This applies (1 / sqrt(N_mc)) * P to aus_vector.
    # Expose the replica structure, center over the sample-within-replica axis,
    # then flatten back to the bare apply_fn output shape expected by the VJP.
    if mode == "complex":
        aus_vector = aus_vector.reshape(n_replicas, ns, 2)
    else:
        aus_vector = aus_vector.reshape(n_replicas, ns)
    aus_vector = (aus_vector - aus_vector.mean(axis=1, keepdims=True)) / jnp.sqrt(N_mc)
    if mode == "complex":
        aus_vector = aus_vector.reshape(N_mc, 2)
    else:
        aus_vector = aus_vector.reshape(N_mc)
    # shape [N_mc // p.size,2]
    if config.netket_sharding:
        aus_vector = jax.lax.with_sharding_constraint(
            aus_vector,
            NamedSharding(
                jax.sharding.get_abstract_mesh(),
                P("S", *(None,) * (aus_vector.ndim - 1)),
            ),
        )

    # _, vjp_fun = jax.vjp(f, parameters_real)
    vjp_fun = nkjax.vjp_chunked(
        _apply_fn,
        parameters_real,
        samples,
        model_state,
        chunk_size=chunk_size,
        chunk_argnums=1,
        nondiff_argnums=(1, 2),
    )

    (updates,) = vjp_fun(aus_vector)  # pytree [N_params,]

    if momentum is not None:
        updates = tree_map(lambda x, y: x + momentum * y, updates, old_updates)
        old_updates = updates

    return rss(updates), old_updates, info
