"""Per-replica statistics container for foundational states.

See ``design/upstream-netket-stats.md`` for the planned netket-side taxonomy
(``StatsBatch`` -> ``MatrixStats``, a generic ``StatsBatch``, and ``ReplicaStats``
as a thin subclass). Until that lands, ``ReplicaStats`` is full-fledged here.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from netket.stats import Stats, statistics
from netket.utils import struct

# Field order of ``netket.stats.Stats`` (used to convert a plain batched Stats
# returned by ``jax.vmap`` into a ``ReplicaStats``).
_FIELDS = ("mean", "error_of_mean", "variance", "tau_corr", "R_hat", "tau_corr_max")


@struct.dataclass
class ReplicaStats(Stats):
    """A batched :class:`netket.stats.Stats`: one entry per replica.

    Every field carries a leading ``(n_replicas,)`` axis, so this is a genuine
    pytree of batched arrays (built by :func:`replica_statistics` via
    ``jax.vmap``) rather than a Python ``list``.

    It still behaves like the old ``list[Stats]``: ``len(rs)`` is the number of
    replicas, ``rs[i]`` and iteration yield the scalar :class:`Stats` of one
    replica (in replica order). The aggregate training-loss summary is exposed
    under :attr:`total`.
    """

    @property
    def shape(self) -> tuple:
        """Shape of the batch — ``(n_replicas,)``."""
        return self.mean.shape

    @property
    def total(self) -> Stats:
        """Aggregate the per-replica statistics into a single summary.

        The combination rule used for the foundational training loss:

        - ``mean``: average of the replica means,
        - ``error_of_mean``: quadrature sum of the replica errors divided by
          the number of replicas (standard error of the grand mean),
        - ``variance``: average of the replica variances,
        - ``tau_corr``: average of the replica autocorrelation times,
        - ``R_hat``: maximum across replicas (the most conservative convergence
          indicator),
        - ``tau_corr_max``: maximum across replicas (the most conservative
          autocorrelation-time estimate).
        """
        n = self.mean.shape[0]
        return Stats(
            mean=jnp.mean(self.mean),
            error_of_mean=jnp.sqrt(jnp.nansum(self.error_of_mean**2)) / n,
            variance=jnp.nansum(self.variance) / n,
            tau_corr=jnp.nanmean(self.tau_corr),
            R_hat=jnp.nanmax(self.R_hat),
            tau_corr_max=jnp.nanmax(self.tau_corr_max),
        )

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, i: int) -> Stats:
        return Stats(*(getattr(self, f)[i] for f in _FIELDS))

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    @classmethod
    def stack(cls, stats: Sequence[Stats]) -> ReplicaStats:
        """Build a :class:`ReplicaStats` by stacking scalar :class:`Stats`.

        Used for the online-statistics path, which produces one
        :class:`Stats` per replica that cannot be obtained via ``vmap``.
        """
        batched = jax.tree.map(lambda *xs: jnp.stack(xs), *stats)
        return cls(*(getattr(batched, f) for f in _FIELDS))

    def __repr__(self):
        inner = ", ".join(repr(s) for s in self)
        return f"ReplicaStats([{inner}], total={self.total!r})"


def replica_statistics(data: jax.Array) -> ReplicaStats:
    """Per-replica :func:`netket.stats.statistics`, vmapped over the replica axis.

    Args:
        data: array of shape ``(n_replicas, n_chains_per_replica, n_samples)``.

    Returns:
        A :class:`ReplicaStats` with one scalar :class:`Stats` worth of fields
        per replica.
    """
    batched = jax.vmap(statistics)(data)
    return ReplicaStats(*(getattr(batched, f) for f in _FIELDS))
