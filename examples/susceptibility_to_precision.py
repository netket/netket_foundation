"""
Minimal example for susceptibility_to_precision.

This script shows how to:
  1. build a small foundational variational state;
  2. extract a physical MCState at one parameter value;
  3. estimate the fidelity susceptibility once with current samples;
  4. keep sampling until a target error bar is reached.

The model below is intentionally tiny and left untrained so the example runs
quickly. For physically meaningful results, train the state first or load a
previously optimized checkpoint.

Usage:
    uv run --with-editable ../netket python examples/susceptibility_to_precision.py
"""

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax.numpy as jnp
import netket as nk
import netket_foundation as nkf

from netket_foundation._src.model.vit import ViTFNQS
from netket_foundation.observable import SusceptibilityObservable

L = 6
hi = nk.hilbert.Spin(0.5, L)
ps = nkf.ParameterSpace(N=1, min=0.8, max=1.2)


model = ViTFNQS(
    num_layers=1,
    d_model=6,
    heads=2,
    L_eff=L // 2,
    n_coups=ps.size,
    b=2,
    complex=False,
    disorder=False,
    transl_invariant=False,
    two_dimensional=False,
)

sampler = nk.sampler.MetropolisLocal(hi, n_chains=64)
vstate = nkf.FoundationalQuantumState(
    sampler,
    model,
    ps,
    n_samples=256,
    n_replicas=4,
    seed=0,
)
vstate.parameter_array = jnp.linspace(0.8, 1.2, vstate.n_replicas).reshape(-1, 1)


h0 = jnp.array([1.0])
mc = vstate.get_state(h0, seed=123)
mc.n_samples = 128
mc.sample()

chi0 = mc.expect(SusceptibilityObservable(hi))
print("[one-shot]")
print(f"  chi = {float(chi0.mean[0, 0]):.5f}")
print(f"  err = {float(chi0.error_of_mean[0, 0]):.5f}")

chi = mc.expect_to_precision(
    SusceptibilityObservable(hi),
    atol=8e-2,
    max_iter=25,
    max_lag=16,
    verbose=True,
).get_stats()
print("\n[to precision]")
print(f"  chi = {float(chi.mean[0, 0]):.5f}")
print(f"  err = {float(chi.error_of_mean[0, 0]):.5f}")
print(f"  converged: {float(jnp.max(chi.error_of_mean)) <= 8e-2}")
