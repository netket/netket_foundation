"""
Fidelity susceptibility of a foundational NQS.

The fidelity susceptibility chi_ij = Cov[d_i log|psi|, d_j log|psi|] is
the quantum geometric tensor diagonal. It peaks at phase transitions and
measures how fast the ground-state manifold curves as parameters change.

This script demonstrates three ways to compute it, all through the
``SusceptibilityObservable``:

  1. mc_state.expect(SusceptibilityObservable(hilbert))
       Direct computation at a single point using existing samples.

  2. mc_state.expect_to_precision(SusceptibilityObservable(hilbert), atol=…)
       Keep drawing new samples until the element-wise error is small enough.

  3. is_state.expect(SusceptibilityObservable(hilbert))
       IS-weighted susceptibility: no additional MCMC needed.

Usage:
    python examples/susceptibility.py
"""

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import jax.numpy as jnp
import netket as nk
import netket_foundation as nkf
import optax

from netket_foundation._src.model.vit import ViTFNQS
from netket_foundation.expectation_value import ISState
from netket_foundation.observable import SusceptibilityObservable

# ---------------------------------------------------------------------------
# System and model
# ---------------------------------------------------------------------------

L = 10
hi = nk.hilbert.Spin(0.5, L)
ps = nkf.ParameterSpace(N=1, min=0.75, max=2.0)


def create_operator(params):
    h = params[0]
    ha_X = sum(nkf.operator.sigmax(hi, i) for i in range(L))
    ha_ZZ = sum(
        nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + 1) % L)
        for i in range(L)
    )
    return -h * ha_X - ha_ZZ


ha_p = nkf.operator.ParametrizedOperator(hi, ps, create_operator)

ma = ViTFNQS(
    num_layers=2,
    d_model=12,
    heads=4,
    L_eff=L // 2,
    n_coups=ps.size,
    b=2,
    complex=False,
    disorder=False,
    transl_invariant=True,
    two_dimensional=False,
)

sa = nk.sampler.MetropolisLocal(hi, n_chains=256)
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_samples=1024, n_replicas=8, seed=1)
vs.parameter_array = jnp.linspace(0.75, 2.0, vs.n_replicas).reshape(-1, 1)

gs = nkf.VMC_NG(ha_p, optax.sgd(0.01), variational_state=vs, diag_shift=1e-2)
print("Training …")
gs.run(200, show_progress=True)

# ---------------------------------------------------------------------------
# Method 1: susceptibility() at a single parameter point
# ---------------------------------------------------------------------------

h0 = jnp.array([1.0])
mc = vs.get_state(h0)
mc.n_samples = 2048
for _ in range(4):
    mc.sample()

chi = mc.expect(SusceptibilityObservable(hi))
print("\n[Method 1] susceptibility at h=1.0")
print(f"  chi = {float(chi.mean[0,0]):.5f}")
print(f"  err = {float(chi.error_of_mean[0,0]):.5f}")

# ---------------------------------------------------------------------------
# Method 2: susceptibility_to_precision — sample until error is small enough
# ---------------------------------------------------------------------------

print("\n[Method 2] susceptibility_to_precision(atol=5e-3)")
mc_fresh = vs.get_state(h0)
mc_fresh.n_samples = 512  # start small; will accumulate batches
mc_fresh.sample()

chi3 = mc_fresh.expect_to_precision(
    SusceptibilityObservable(hi), atol=5e-3, verbose=True
).get_stats()
print(f"  chi = {float(chi3.mean[0,0]):.5f}")
print(f"  err = {float(chi3.error_of_mean[0,0]):.5f}")
print(f"  converged: {float(jnp.max(chi3.error_of_mean)) <= 5e-3}")

# ---------------------------------------------------------------------------
# Method 3: IS-weighted susceptibility from a nearby reference
# ---------------------------------------------------------------------------

print("\n[Method 3] IS susceptibility from h=0.95 reference")
h_ref = jnp.array([0.95])
mc_ref = vs.get_state(h_ref)
mc_ref.n_samples = 4096
for _ in range(8):
    mc_ref.sample()

# Target at h=1.0, reweighting mc_ref's samples
is_st = ISState.from_mc_state(mc_ref, h0)

chi4 = is_st.expect(SusceptibilityObservable(hi))
print(f"  chi = {float(chi4.mean[0,0]):.5f}")
print(f"  err = {float(chi4.error_of_mean[0,0]):.5f}")
print(f"  ESS = {is_st.ess:.0f} ({is_st.ess_fraction:.1%})")

# ---------------------------------------------------------------------------
# Sweep: susceptibility profile across h range
# ---------------------------------------------------------------------------

print("\nSweeping chi(h) from h=0.75 to h=2.0 …")

# One anchor per sweep region; IS reweights to each target h.
h_anchors = [0.85, 1.1, 1.5, 1.9]
anchor_states = {}
for h_a in h_anchors:
    mc = vs.get_state(jnp.array([h_a]))
    mc.n_samples = 4096
    for _ in range(8):
        mc.sample()
    anchor_states[h_a] = mc

h_sweep = np.linspace(0.75, 2.0, 41)
chi_vals = []

for h0 in h_sweep:
    h_anc = h_anchors[np.argmin(np.abs(np.array(h_anchors) - h0))]
    mc_ref = anchor_states[h_anc]
    pars = jnp.array([h0])

    is_st = ISState.from_mc_state(mc_ref, pars)
    result = is_st.expect(SusceptibilityObservable(hi))
    chi_vals.append(float(result.mean[0, 0]))

h_peak = h_sweep[np.argmax(chi_vals)]
print(f"  chi peak at h ≈ {h_peak:.2f}  (exact: h_c = 1.0 in thermodynamic limit)")
