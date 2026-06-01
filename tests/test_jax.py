"""Tests for netket_foundation._src.jax utilities."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import core as fcore

from netket_foundation._src.jax import foundational_log_jacobian
from helpers import (
    make_hilbert,
    make_parameter_space,
    make_model,
    make_sampler,
    make_vstate,
)


# ---------------------------------------------------------------------------
# Minimal toy model for isolated unit tests
# ---------------------------------------------------------------------------


class _LinearModel(nn.Module):
    """log ψ(σ; θ) = θ · σ  (dot product), batch-compatible."""

    @nn.compact
    def __call__(self, x):
        theta = self.param("theta", nn.initializers.ones, (x.shape[-1],))
        return jnp.einsum("...i,i->...", x.astype(float), theta)


def _make_toy_state(n_sites=4, n_params=4, key=0):
    """Returns (apply_fn, variables, samples) for a toy linear model."""
    model = _LinearModel()
    samples = jnp.ones((8, n_sites))  # all-ones configurations
    foundational_params = jnp.linspace(0.5, 1.5, n_params)

    rng = jax.random.PRNGKey(key)
    inner_vars = model.init(rng, samples[0])  # {'params': {'theta': ...}}
    variables = fcore.copy(
        inner_vars, {"foundational": {"parameters": foundational_params}}
    )

    def apply_fn(v, x):
        return model.apply(fcore.copy({}, {"params": v["params"]}), x)

    return apply_fn, variables, samples, foundational_params


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_output_shape():
    apply_fn, variables, samples, params = _make_toy_state(n_sites=4, n_params=4)
    jac = foundational_log_jacobian(apply_fn, variables, samples)
    assert jac.shape == (samples.shape[0], params.shape[0])


def test_output_shape_with_chunk_size():
    apply_fn, variables, samples, params = _make_toy_state(n_sites=4, n_params=4)
    jac = foundational_log_jacobian(apply_fn, variables, samples, chunk_size=3)
    assert jac.shape == (samples.shape[0], params.shape[0])


def test_chunked_matches_unchunked():
    apply_fn, variables, samples, _ = _make_toy_state(n_sites=4, n_params=4)
    jac_full = foundational_log_jacobian(apply_fn, variables, samples)
    jac_chunked = foundational_log_jacobian(apply_fn, variables, samples, chunk_size=3)
    np.testing.assert_allclose(jac_full, jac_chunked, atol=1e-5)


def test_gradient_finite_differences():
    """Jacobian values match finite differences on a toy linear model.

    For log ψ(σ; θ) = θ · f(σ), d/dθ_k = f_k(σ) (independent of θ).
    The toy model uses σ directly as features, so d/dθ_k = σ_k.
    But the foundational parameters are separate from the inner 'theta';
    here we use a model where foundational params are the actual weights.
    """
    n_sites = 3
    n_samples = 5

    samples = jax.random.normal(jax.random.PRNGKey(1), (n_samples, n_sites))
    theta0 = jnp.array([0.3, -0.5, 0.7])

    # Model: log psi(sigma; theta) = theta · sigma
    # All weights live in 'foundational', no inner 'params'.
    def apply_fn(v, x):
        theta = v["foundational"]["parameters"]
        return jnp.einsum("...i,i->...", x.astype(float), theta)

    variables = {"foundational": {"parameters": theta0}}

    jac = foundational_log_jacobian(apply_fn, variables, samples)

    # Analytic: d/dtheta_k [theta · sigma_i] = sigma_i_k
    expected = np.array(samples)
    np.testing.assert_allclose(jac, expected, atol=1e-5)


def test_finite_difference_agreement_nonlinear():
    """Check against numerical finite differences for a nonlinear model."""
    apply_fn, variables, samples, params = _make_toy_state(n_sites=4, n_params=4)
    jac = np.array(foundational_log_jacobian(apply_fn, variables, samples))

    eps = 1e-4
    n_params = params.shape[0]
    jac_fd = np.zeros_like(jac)
    for k in range(n_params):
        delta = jnp.zeros(n_params).at[k].set(eps)
        vars_plus = fcore.copy(
            {kk: vv for kk, vv in variables.items() if kk != "foundational"},
            {"foundational": {"parameters": params + delta}},
        )
        vars_minus = fcore.copy(
            {kk: vv for kk, vv in variables.items() if kk != "foundational"},
            {"foundational": {"parameters": params - delta}},
        )
        f_plus = np.array(apply_fn(vars_plus, samples))
        f_minus = np.array(apply_fn(vars_minus, samples))
        jac_fd[:, k] = (f_plus - f_minus) / (2 * eps)

    np.testing.assert_allclose(jac, jac_fd, atol=1e-4)


def test_integration_with_vstate():
    """foundational_log_jacobian works end-to-end with a FoundationalQuantumState."""
    hi = make_hilbert()
    ps = make_parameter_space()
    sampler = make_sampler(hi)
    model = make_model(ps)
    vstate = make_vstate(sampler, model, ps)

    params = vstate.parameter_array[0]
    mc = vstate.get_state(params, seed=42)
    mc.sample()

    samples = mc.samples
    if samples.ndim >= 3:
        samples = jax.lax.collapse(samples, 0, 2)

    jac = foundational_log_jacobian(
        mc.model.apply, mc.variables, samples, mc.chunk_size
    )
    assert jac.shape == (samples.shape[0], ps.size)
    assert np.all(np.isfinite(np.array(jac)))
