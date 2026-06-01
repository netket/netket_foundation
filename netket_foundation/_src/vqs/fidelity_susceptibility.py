"""
Fidelity susceptibility chi_ij = Re Cov[d_i log psi, d_j log psi].

The core object is SusceptibilityObservable, which plugs into the standard
netket LocalEstimator interface:

    obs = SusceptibilityObservable(vstate.hilbert)
    vstate.expect(obs)                             # StatsBatch
    vstate.expect_to_precision(obs, atol=1e-3)     # OnlineStatsBatch accumulator

It is also dispatched on by ISState.expect() for importance-sampled estimates.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from netket.jax import HashablePartial
from netket.operator._abstract_observable import AbstractObservable
from netket.stats import LocalEstimatorsBatch
from netket.vqs import (
    MCState,
    expect as nk_expect,
    local_estimators as nk_local_estimators,
)

from netket_foundation._src.jax import foundational_log_jacobian
from netket_foundation._src.expectation_value.importance_sampling import ISMatrixResult
from netket_foundation._src.expectation_value.is_state import ISState


class SusceptibilityObservable(AbstractObservable):
    """Observable for the fidelity susceptibility matrix.

    Registers with NetKet's LocalEstimator dispatch so that
    ``vstate.expect(obs)`` and ``vstate.expect_to_precision(obs, ...)``
    work out of the box.

    Args:
        hilbert: Hilbert space of the variational state (``vstate.hilbert``).
    """


# ---------------------------------------------------------------------------
# Dispatches
# ---------------------------------------------------------------------------


@nk_expect.dispatch
def _susceptibility_expect(
    vstate: MCState,
    observable: SusceptibilityObservable,
    chunk_size: int | None,
):
    return vstate.local_estimators(observable, chunk_size=chunk_size).to_stats()


@nk_local_estimators.dispatch
def _susceptibility_local_estimators(
    vstate: MCState,
    observable: SusceptibilityObservable,
    chunk_size: int | None,
) -> LocalEstimatorsBatch:
    variables = vstate.variables

    if "foundational" not in variables:
        raise ValueError(
            "SusceptibilityObservable requires the variational state to expose a "
            "'foundational' collection in its variables (the parameters w.r.t. which "
            "the fidelity susceptibility is computed), but the variables only contain "
            f"the collections {tuple(variables.keys())}."
        )

    sigma = vstate.samples
    n_chains = sigma.shape[0]
    if sigma.ndim >= 3:
        sigma = jax.lax.collapse(sigma, 0, 2)

    dlog = foundational_log_jacobian(vstate.model.apply, variables, sigma, chunk_size)
    n_params = dlog.shape[-1]
    channels = jnp.concatenate(
        [
            dlog,
            (jnp.conj(dlog[:, :, None]) * dlog[:, None, :]).reshape(dlog.shape[0], -1),
        ],
        axis=-1,
    )
    return LocalEstimatorsBatch(
        data=channels.reshape(n_chains, -1, n_params + n_params * n_params),
        combinator=HashablePartial(_combine, n_params),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _combine(n_params: int, mu):
    mean_dlog = mu[:n_params]
    mean_outer = mu[n_params:].reshape(n_params, n_params)
    qgt = mean_outer - jnp.conj(mean_dlog)[:, None] * mean_dlog[None, :]
    return jnp.real(qgt)


# ---------------------------------------------------------------------------
# IS overload — registers with ISState.expect dispatch
# ---------------------------------------------------------------------------


@nk_expect.dispatch
def expect_isstate_susceptibility(
    is_state: ISState,
    observable: SusceptibilityObservable,
    chunk_size: int | None,
) -> ISMatrixResult:
    if "foundational" not in is_state.variables:
        raise ValueError(
            "SusceptibilityObservable requires the target state to expose a "
            "'foundational' collection in its variables (the parameters w.r.t. which "
            "the fidelity susceptibility is computed), but the variables only contain "
            f"the collections {tuple(is_state.variables.keys())}."
        )

    return _is_susceptibility_result(
        is_state.apply_fn,
        is_state.variables,
        is_state.samples,
        is_state.normalized_weights,  # (n_samples,) — cached
        is_state.ess,
        is_state.ess_fraction,
        chunk_size or is_state._chunk_size,
    )


@partial(jax.jit, static_argnames=("apply_fn", "chunk_size"))
def _is_susceptibility_result(
    apply_fn, target_variables, samples, w, ess, ess_fraction, chunk_size
) -> ISMatrixResult:
    """IS-weighted fidelity-susceptibility estimate, packaged as ISMatrixResult.

    Args:
        apply_fn:         Target model apply function.
        target_variables: Variables for the target state.
        samples:          Reference samples, shape (n_samples, N).
        w:                Normalized IS weights, shape (n_samples,).
        ess:              Effective sample size.
        ess_fraction:     ESS as a fraction of n_samples.
        chunk_size:       Optional chunk size for the Jacobian evaluation.
    """
    dlog = foundational_log_jacobian(
        apply_fn, target_variables, samples, chunk_size
    )  # (n_samples, n_params)

    n = dlog.shape[0]
    mu = jnp.einsum("i,ij->j", w, dlog)
    d = dlog - mu[None, :]

    # chi_ij is the real part of the Hermitian IS-weighted covariance.
    chi_local = jnp.conj(d[:, :, None]) * d[:, None, :]
    chi_mean = jnp.real(jnp.einsum("i,ijk->jk", w, chi_local))

    # Standard error of the self-normalized IS estimator via the delta method,
    # sqrt(sum_i w_i^2 (chi_local_i - chi_mean)^2), applied element-wise — same
    # estimator as the scalar ISResult path. Exact (asymptotically) for
    # arbitrary weights and reduces to the plain MC error sqrt(var/n) when the
    # weights are uniform (w_i = 1/n, ESS == n).
    sq_dev = jnp.abs(chi_local - chi_mean[None]) ** 2
    error_of_mean = jnp.sqrt(jnp.einsum("i,ijk->jk", w**2, sq_dev))

    return ISMatrixResult(
        mean=chi_mean,
        error_of_mean=error_of_mean,
        ess=ess,
        ess_fraction=ess_fraction,
        n_samples=n,
    )
