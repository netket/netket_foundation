"""
Result types and shared low-level utilities for importance-sampled estimators.

IS identity:
    <O>_target = sum_i w_i * O_loc_target(sigma_i) / sum_i w_i
    w_i = |psi_target(sigma_i)|^2 / |psi_ref(sigma_i)|^2
    sigma_i ~ |psi_ref|^2
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from netket.stats import Stats, StatsBatch
from netket.utils import struct


@struct.dataclass
class ISResult(Stats):
    """:class:`netket.stats.Stats` extended with importance-sampling diagnostics
    (:attr:`ess`, :attr:`ess_fraction`, :attr:`n_samples`)."""

    ess: float = float("nan")
    """Effective sample size."""
    ess_fraction: float = float("nan")
    """ESS as a fraction of the total number of reference samples."""
    n_samples: int = 0
    """Number of reference samples used."""

    def to_dict(self):
        jsd = super().to_dict()
        jsd["ESS"] = self.ess
        jsd["ESSFraction"] = self.ess_fraction
        return jsd

    def __repr__(self):
        # Reuse Stats formatting for mean ± σ [var=…] then append ESS.
        base = super().__repr__()  # ends with ']'
        return base[:-1] + f", ESS={self.ess:.0f} ({self.ess_fraction:.1%})]"


@struct.dataclass
class ISMatrixResult(StatsBatch):
    """:class:`netket.stats.StatsBatch` extended with importance-sampling
    diagnostics (:attr:`ess`, :attr:`ess_fraction`, :attr:`n_samples`)."""

    ess: float = float("nan")
    """Effective sample size."""
    ess_fraction: float = float("nan")
    """ESS as a fraction of the total number of reference samples."""
    n_samples: int = 0
    """Number of reference samples used."""

    def __repr__(self):
        err_max = float(jnp.max(jnp.abs(self.error_of_mean)))
        return (
            f"ISMatrixResult(shape={self.shape}, max_err={err_max:.4g},"
            f" ESS={self.ess:.0f} ({self.ess_fraction:.1%}))"
        )


# ---------------------------------------------------------------------------
# Internal utility — used by nk_expect overloads registered on ISState
# ---------------------------------------------------------------------------


class ISWeights(NamedTuple):
    """Normalized importance-sampling weights and their effective sample size."""

    normalized_weights: jax.Array
    """IS weights normalized to sum to 1, shape (n_samples,)."""
    ess: float
    """Effective sample size."""


def _is_weights(log_prob_target, log_prob_ref) -> ISWeights:
    """
    Compute normalized IS weights and ESS from log-probabilities.

    Args:
        log_prob_target: shape (n,), log|psi_target(sigma)|^2
        log_prob_ref:    shape (n,), log|psi_ref(sigma)|^2

    Returns:
        :class:`ISWeights` — ``normalized_weights`` (summing to 1) and ``ess``.
    """
    log_w = log_prob_target - log_prob_ref
    log_w = log_w - log_w.max()  # stabilize
    w = jnp.exp(log_w)
    W = w.sum()
    # Keep ESS as a (0-d) array: forcing a Python float here would trigger a
    # device-to-host sync (and a gather under sharding). jnp.sum already inserts
    # the global cross-device reduction, so the array value is correct as-is.
    ess = W**2 / (w**2).sum()
    return ISWeights(normalized_weights=w / W, ess=ess)
