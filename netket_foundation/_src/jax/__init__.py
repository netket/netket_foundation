"""
Low-level JAX utilities for foundational-parameter models.

These helpers operate on Flax variable trees that contain a ``'foundational'``
collection with a ``'parameters'`` key — the convention used throughout
netket_foundation.
"""

from __future__ import annotations

import jax
from flax import core as fcore
from netket import jax as nkjax


def foundational_log_jacobian(
    apply_fn,
    variables,
    samples,
    chunk_size=None,
):
    """
    Jacobian of log ψ with respect to the foundational parameters.

    Computes  J[i, k] = d log ψ(σ_i) / d θ_k  where θ are the entries of
    ``variables['foundational']['parameters']`` and σ_i are the rows of
    ``samples``.

    The ``'foundational'`` collection is isolated from the full variable
    tree and differentiated independently; all other collections (e.g.
    ``'params'``) are treated as frozen constants.  Forward-mode AD
    (``jax.jacfwd``) is used because the number of parameters is typically
    much smaller than the number of samples.

    Args:
        apply_fn:   ``model.apply`` callable with signature
                    ``apply_fn(variables, samples) -> log_amplitudes``.
                    ``log_amplitudes`` must have shape ``(n_samples,)``.
        variables:  Flax variable dict containing a ``'foundational'``
                    collection with a ``'parameters'`` key of shape
                    ``(n_params,)``.
        samples:    Array of shape ``(n_samples, hilbert_size)``.
        chunk_size: When given, the Jacobian is evaluated in chunks of
                    this size along the sample axis, reducing peak memory
                    at the cost of more sequential computation.  ``None``
                    evaluates the full batch in a single call.

    Returns:
        Array of shape ``(n_samples, n_params)`` with dtype matching the
        output of ``apply_fn``.
    """
    vars_no_h, h_dict = fcore.pop(variables, "foundational")

    def logpsi_h(h, x):
        return apply_fn(fcore.copy(vars_no_h, {"foundational": h}), x)

    if chunk_size is None:
        df = jax.jacfwd(logpsi_h)(h_dict, samples)
    else:
        jac_chunked = nkjax.apply_chunked(
            jax.jacfwd(logpsi_h), in_axes=(None, 0), chunk_size=chunk_size
        )
        jac_chunked = jax.jit(jac_chunked)
        df = jac_chunked(h_dict, samples)

    return df["parameters"]  # (n_samples, n_params)
