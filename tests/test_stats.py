"""ReplicaStats container and the per-replica combination rule."""

import jax
import jax.numpy as jnp
import numpy as np

from netket.stats import Stats, statistics

from netket_foundation.stats import ReplicaStats, replica_statistics


def _make_stats(mean, err, var, rhat):
    return Stats(
        mean=jnp.asarray(mean),
        error_of_mean=jnp.asarray(err),
        variance=jnp.asarray(var),
        R_hat=jnp.asarray(rhat),
    )


def _replica_stats():
    return ReplicaStats.stack(
        [_make_stats(1.0, 0.3, 1.0, 1.01), _make_stats(3.0, 0.4, 3.0, 1.20)]
    )


def test_replica_stats_is_list_like():
    stats = _replica_stats()
    # It is a (batched) Stats, no longer a Python list, but behaves like one.
    assert isinstance(stats, Stats)
    assert not isinstance(stats, list)
    assert len(stats) == 2
    assert stats.shape == (2,)
    # indexing yields the scalar Stats of one replica
    assert isinstance(stats[0], Stats)
    assert float(stats[0].mean) == 1.0
    # iteration yields per-replica scalar Stats in order
    means = [float(s.mean) for s in stats]
    np.testing.assert_allclose(means, [1.0, 3.0])


def test_total_combination_rule():
    stats = _replica_stats()
    total = stats.total
    np.testing.assert_allclose(float(total.mean), 2.0)
    np.testing.assert_allclose(float(total.error_of_mean), np.sqrt(0.3**2 + 0.4**2) / 2)
    np.testing.assert_allclose(float(total.variance), 2.0)
    np.testing.assert_allclose(float(total.R_hat), 1.20)


def test_replica_statistics_matches_per_replica():
    """replica_statistics == vmap of statistics over the replica axis."""
    data = jax.random.normal(jax.random.PRNGKey(0), (3, 4, 50))
    rs = replica_statistics(data)
    assert isinstance(rs, ReplicaStats)
    assert rs.shape == (3,)
    for i in range(3):
        ref = statistics(data[i])
        np.testing.assert_allclose(rs[i].mean, ref.mean, rtol=1e-6)
        np.testing.assert_allclose(rs[i].error_of_mean, ref.error_of_mean, rtol=1e-6)
        np.testing.assert_allclose(rs[i].R_hat, ref.R_hat, rtol=1e-6, equal_nan=True)


def test_replica_stats_is_a_pytree():
    stats = _replica_stats()
    doubled = jax.tree.map(lambda x: 2 * x, stats)
    assert isinstance(doubled, ReplicaStats)
    np.testing.assert_allclose(float(doubled[1].mean), 6.0)
    # survives a jit boundary
    out = jax.jit(lambda s: s)(stats)
    assert isinstance(out, ReplicaStats)
