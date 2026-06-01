"""
Save / load variational states bundled with their importance-sampling reference.

:func:`save` writes an nqxpack archive containing the state together with its
sampled configurations and reference log-probabilities (a :class:`SamplesWithProb`
bundle), so that loading is instantaneous — no re-thermalisation and no
re-sampling — and the IS reference is exactly reproducible.

Works under distributed (sharded) execution: sharded arrays are gathered before
writing, and re-sharded along the sample axis ``"S"`` on load.
"""

from __future__ import annotations

import jax

import nqxpack

from netket.hilbert import TensorHilbert

from netket_foundation._src.expectation_value.samples_with_probability import (
    SamplesWithProb as SamplesWithProb,
)
from netket_foundation._src.hilbert.parameter_space import ParameterSpace


def samples_with_probability(state) -> SamplesWithProb:
    """Return the reference :class:`SamplesWithProb` of ``state``.

    The samples and log-density ``log p_ref = machine_pow * log|psi_ref|`` are
    returned as (sharded) JAX arrays — they are not gathered to a single process.
    The sampling power is read from the state and folded into ``log_probs``, so
    the bundle is self-contained (the IS weights only need the density values).

    When the state samples a joint ``physical ⊗ ParameterSpace`` space (a
    :class:`FoundationalQuantumState`), the stored ``.samples`` carry the
    per-replica parameter columns. The log-density is evaluated on those joint
    samples (so each row uses its own replica parameters), but only the
    *physical* columns are kept in the bundle — that is what the physical target
    model in an :class:`ISState` consumes. For a plain ``MCState`` the samples
    are already physical and are used as-is.
    """
    samples = state.samples
    if samples.ndim >= 3:
        samples = jax.lax.collapse(samples, 0, 2)

    log_probs = state.sampler.machine_pow * state.log_value(samples).real

    # Joint physical ⊗ ParameterSpace state: drop the trailing parameter columns.
    hilb = state.hilbert
    if isinstance(hilb, TensorHilbert) and isinstance(
        hilb.subspaces[-1], ParameterSpace
    ):
        n_physical = hilb.size - hilb.subspaces[-1].size
        samples = samples[:, :n_physical]

    return SamplesWithProb(samples, log_probs)


def save(path: str, state, *, key: str = "state") -> None:
    """Save ``state`` together with its reference samples and log-probabilities.

    The archive contains ``{key: state, "reference": SamplesWithProb(...)}``, so
    it can later be turned into an :class:`~netket_foundation.expectation_value.ISState`
    via :meth:`ISState.from_nqxpack` without re-sampling.

    Args:
        path:  Destination ``.nk`` path.
        state: A sampled variational state (MCState / FoundationalQuantumState).
        key:   Key under which the state is stored (default ``"state"``).
    """
    reference = samples_with_probability(state)
    nqxpack.save({key: state, "reference": reference}, path)


def load(path: str, *, return_dict: bool = False, key: str = "state"):
    """Load an archive written by :func:`save`.

    Returns the state by default, or the full ``{key, "reference"}`` dict when
    ``return_dict=True``.
    """
    data = nqxpack.load(path)
    return data if return_dict else data[key]
