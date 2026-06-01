"""
Importance-sampling sweep over a parameter range.

After training a foundational NQS, IS lets you evaluate observables across
a dense grid of parameter values without re-running MCMC at each point.
The key object is ISState: it wraps a fixed reference sample set and caches
the IS weights so that multiple observables share one weight computation.

This script:
  1. Trains a small FNQS on the 1D transverse-field Ising model.
  2. Samples a handful of "anchor" MCStates across the h range.
  3. Sweeps 51 h values via IS, computing energy AND susceptibility
     per point with shared weights (one weight computation per target).
  4. Saves anchor states to disk and loads them back via ISState.from_nqxpack.
  5. Compares the ESS across anchors to show how to pick the best reference.

Usage:
    python examples/is_sweep.py
"""

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import time
import numpy as np
import jax.numpy as jnp
import netket as nk
import netket_foundation as nkf
import optax
from tqdm import tqdm

from netket_foundation._src.model.vit import ViTFNQS
from netket_foundation.expectation_value import (
    ISState,
    SamplesWithProb,
)
from netket_foundation.observable import SusceptibilityObservable
from netket_foundation.vqs import samples_with_probability

# ---------------------------------------------------------------------------
# System
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


Mz = sum(nkf.operator.sigmaz(hi, i) for i in range(L)) * (1.0 / L)
Mz2 = Mz @ Mz  # type: ignore[operator]

ha_p = nkf.operator.ParametrizedOperator(hi, ps, create_operator)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

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
t0 = time.perf_counter()
gs.run(200, show_progress=True)
print(f"Done in {time.perf_counter() - t0:.0f}s")

# ---------------------------------------------------------------------------
# Sample anchor MCStates and save them to disk
# ---------------------------------------------------------------------------

h_anchors = np.linspace(0.8, 1.9, 7)
h_sweep = np.linspace(0.75, 2.0, 51)

os.makedirs("anchor_refs", exist_ok=True)
anchor_states: dict[float, nk.vqs.MCState] = {}

for h0 in h_anchors:
    mc = vs.get_state(jnp.array([h0]))
    mc.n_samples = 4096
    for _ in range(8):  # burn-in
        mc.sample()
    anchor_states[h0] = mc
    # Save as nqxpack (state + reference bundled) for demo of from_nqxpack
    nkf.vqs.save(f"anchor_refs/h{h0:.2f}.nk", mc)
    # Also save as .npz (raw arrays, no model needed to reload)
    samples_with_probability(mc).save(f"anchor_refs/h{h0:.2f}.npz")
    print(f"  anchor h={h0:.2f}  samples={mc.samples.shape}")


def nearest_anchor(h):
    return h_anchors[np.argmin(np.abs(h_anchors - h))]


# ---------------------------------------------------------------------------
# IS sweep — energy + Mz2 + susceptibility per point, shared weights
# ---------------------------------------------------------------------------

results = []
for h0 in tqdm(h_sweep, desc="IS sweep"):
    h_anc = nearest_anchor(h0)
    mc_ref = anchor_states[h_anc]
    pars = jnp.array([h0])

    # Build ISState: weights computed lazily and cached once per (ref, target).
    is_st = ISState.from_mc_state(mc_ref, pars)

    # --- energy and magnetisation share cached weights ---
    r_E = is_st.expect(create_operator(pars))
    r_Mz2 = is_st.expect(Mz2)

    # --- fidelity susceptibility also reuses cached weights ---
    r_chi = is_st.expect(SusceptibilityObservable(hi))

    results.append(
        {
            "h": h0,
            "E": float(r_E.mean.real),
            "E_err": float(r_E.error_of_mean),
            "Mz2": float(r_Mz2.mean.real),
            "chi": float(r_chi.mean[0, 0]),
            "chi_err": float(r_chi.error_of_mean[0, 0]),
            "ess": is_st.ess,
            "ess_frac": is_st.ess_fraction,
            "anchor": h_anc,
        }
    )

print("\nIS sweep done.")
print(f"  Mean ESS fraction: {np.mean([r['ess_frac'] for r in results]):.2%}")
print(f"  Chi peak at h = {results[np.argmax([r['chi'] for r in results])]['h']:.2f}")

# ---------------------------------------------------------------------------
# Demo: load reference back from .npz (model-free) and compare ESS
# ---------------------------------------------------------------------------

print("\nComparing ESS for different .npz references at h=1.0 …")
# Raw-array constructor: supply the target model (apply_fn) and its variables.
target = vs.get_state(jnp.array([1.0]))
apply_fn, target_vars = target._apply_fun, target.variables

for h_ref in h_anchors:
    samples, log_probs = SamplesWithProb.load(f"anchor_refs/h{h_ref:.2f}.npz")
    is_st = ISState(samples, log_probs, apply_fn, target_vars)
    print(f"  ref h={h_ref:.2f} → ESS={is_st.ess:.0f} ({is_st.ess_fraction:.1%})")

# ---------------------------------------------------------------------------
# Demo: load anchor back from nqxpack file
# ---------------------------------------------------------------------------

print("\nLoading anchor nearest to h=1.0 from nqxpack …")
h_near = nearest_anchor(1.0)
is_nqx = ISState.from_nqxpack(f"anchor_refs/h{h_near:.2f}.nk", jnp.array([1.0]))
print(f"  ESS from nqxpack file (ref h={h_near:.2f}): {is_nqx.ess:.0f}")
