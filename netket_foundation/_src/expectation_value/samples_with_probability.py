from __future__ import annotations

import os

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils

from nqxpack._src.lib_v1.custom_types import register_serialization
from nqxpack._src.contextmgr import current_context

from netket.jax import sharding


class SamplesWithProb:
    """A bundle of sampled configurations and their reference log-probabilities.

    Attributes:
        samples:   Physical configurations, shape ``(n_samples, N)``.
        log_probs: Reference log-density ``log p_ref(sigma)`` at each sample,
                   shape ``(n_samples,)``.

    The arrays are kept as sharded JAX arrays. Use :meth:`save` /
    :meth:`load` to round-trip through a lightweight ``.npz`` file.
    """

    def __init__(self, samples, log_probs):
        self.samples = samples
        self.log_probs = log_probs

    def __repr__(self):
        return f"SamplesWithProb(n_samples={np.shape(self.samples)[0]})"

    def __iter__(self):
        # Allow ``samples, log_probs = bundle`` tuple-unpacking.
        yield self.samples
        yield self.log_probs

    def save(self, path: str) -> None:
        """Save the bundle to a ``.npz`` file at ``path``.

        Sharded arrays are gathered to fully-addressable host arrays; only the
        main process touches the disk.
        """
        samples = sharding.extract_replicated(sharding.gather(self.samples))
        log_probs = sharding.extract_replicated(sharding.gather(self.log_probs))

        if jax.process_index() == 0:
            np.savez(path, samples=np.asarray(samples), log_probs=np.asarray(log_probs))

        # Ensure the file is fully written before any process proceeds.
        if jax.process_count() > 1:
            multihost_utils.sync_global_devices("SamplesWithProb.save")

    @classmethod
    def load(cls, path: str) -> SamplesWithProb:
        """Load a bundle written by :meth:`save`.

        Both arrays are sharded along the sample axis ``"S"`` so they are ready
        for use in a distributed :class:`ISState`.
        """
        if not os.path.exists(path) and not path.endswith(".npz"):
            path = path + ".npz"  # np.savez appends .npz; np.load does not
        data = np.load(path)
        samples = sharding.shard_along_axis(jnp.asarray(data["samples"]), axis=0)
        log_probs = sharding.shard_along_axis(jnp.asarray(data["log_probs"]), axis=0)
        return cls(samples, log_probs)


# ---------------------------------------------------------------------------
# nqxpack serialization (used when a SamplesWithProb is bundled into an archive)
# ---------------------------------------------------------------------------


def _serialize_samples_with_prob(obj: SamplesWithProb) -> dict:
    # Arrays are written as a msgpack asset (the JSON object tree cannot hold
    # raw array bytes); sharded arrays are gathered and only the main process
    # touches the disk.
    samples = sharding.extract_replicated(sharding.gather(obj.samples))
    log_probs = sharding.extract_replicated(sharding.gather(obj.log_probs))
    current_context().asset_manager.write_msgpack(
        "reference.msgpack",
        {"samples": np.asarray(samples), "log_probs": np.asarray(log_probs)},
    )
    return {}


def _deserialize_samples_with_prob(obj: dict) -> SamplesWithProb:
    data = current_context().asset_manager.read_msgpack("reference.msgpack")
    samples = sharding.shard_along_axis(jnp.asarray(data["samples"]), axis=0)
    log_probs = sharding.shard_along_axis(jnp.asarray(data["log_probs"]), axis=0)
    return SamplesWithProb(samples, log_probs)


register_serialization(
    SamplesWithProb,
    _serialize_samples_with_prob,
    _deserialize_samples_with_prob,
)
