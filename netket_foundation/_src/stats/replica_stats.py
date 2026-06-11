"""Per-replica statistics container for foundational states."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from netket.stats import Stats


def combine_replica_stats(stats: Sequence[Stats]) -> Stats:
    """Combine per-replica :class:`netket.stats.Stats` into a single summary.

    The combination rule used for the foundational training loss:

    - ``mean``: average of the replica means,
    - ``error_of_mean``: quadrature sum of the replica errors divided by the
      number of replicas (standard error of the grand mean),
    - ``variance``: average of the replica variances,
    - ``tau_corr``: average of the replica autocorrelation times,
    - ``R_hat``: maximum across replicas (the most conservative convergence
      indicator),
    - ``tau_corr_max``: maximum across replicas (the most conservative
      autocorrelation-time estimate).
    """
    n = len(stats)
    means = jnp.array([s.mean for s in stats])
    errors = jnp.array([s.error_of_mean for s in stats])
    variances = jnp.array([s.variance for s in stats])
    tau_corrs = jnp.array([s.tau_corr for s in stats])
    rhats = jnp.array([s.R_hat for s in stats])
    tau_corr_maxs = jnp.array([s.tau_corr_max for s in stats])
    return Stats(
        mean=jnp.mean(means),
        error_of_mean=jnp.sqrt(jnp.nansum(errors**2)) / n,
        variance=jnp.nansum(variances) / n,
        tau_corr=jnp.nanmean(tau_corrs),
        R_hat=jnp.nanmax(rhats),
        tau_corr_max=jnp.nanmax(tau_corr_maxs),
    )


class ReplicaStats(list):
    """A list of per-replica :class:`netket.stats.Stats` with a ``.total``.

    Behaves exactly like ``list[Stats]`` (one entry per replica, in replica
    order) and additionally exposes the aggregate statistics under
    :attr:`total`, combined with the same rule used for the training loss
    (see :func:`combine_replica_stats`).

    Registered as a JAX pytree (with the same flattening as ``list``), so it
    passes transparently through ``jax.jit`` boundaries.
    """

    @property
    def total(self) -> Stats:
        return combine_replica_stats(self)

    def __repr__(self):
        inner = ", ".join(repr(s) for s in self)
        return f"ReplicaStats([{inner}], total={self.total!r})"


jax.tree_util.register_pytree_node(
    ReplicaStats,
    lambda x: (list(x), None),
    lambda _, children: ReplicaStats(children),
)
