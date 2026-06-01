"""Tests for the importance-sampling pipeline: ISState, SamplesWithProb, IO.

These cover the regression where ``samples_with_probability`` on a
FoundationalQuantumState returned the *joint* samples (physical + parameter
columns), which the physical target model could not consume — so every
disk-backed reference (``.npz`` / nqxpack) was broken.
"""

import os

import numpy as np
import pytest

import netket_foundation as nkf
from netket_foundation.expectation_value import (
    ISState,
    ISResult,
    ISMatrixResult,
    SamplesWithProb,
)
from helpers import (
    make_hilbert,
    make_parameter_space,
    make_sampler,
    make_model,
    make_ising,
    make_vstate,
)


@pytest.fixture(scope="module")
def vstate():
    hi = make_hilbert()
    ps = make_parameter_space()
    sampler = make_sampler(hi)
    model = make_model(ps)
    vs = make_vstate(sampler, model, ps)
    vs.sample()
    return vs


@pytest.fixture(scope="module")
def obs(vstate):
    return nkf.observable.SusceptibilityObservable(vstate.hilbert_physical)


# ---------------------------------------------------------------------------
# Regression: physical-sample stripping
# ---------------------------------------------------------------------------


def test_samples_with_probability_strips_parameter_columns(vstate):
    """For a FoundationalQuantumState the bundle must hold *physical* samples
    only (parameter columns stripped), with one log-prob per sample."""
    ref = nkf.vqs.samples_with_probability(vstate)
    n_joint = int(np.prod(vstate.samples.shape[:-1]))
    assert ref.samples.shape == (n_joint, vstate.hilbert_physical.size)
    assert ref.log_probs.shape == (n_joint,)


def test_reference_routes_agree(vstate, obs, tmp_path):
    """All four ways of building the reference must produce identical results.

    Before the fix, the disk/SamplesWithProb routes fed joint-width samples to
    the physical target model — a crash (width-sensitive model) or silently
    wrong numbers. This test pins the routes together.
    """
    p0 = vstate.parameter_array[0]

    a = vstate.is_state(p0)  # reference=None (current samples)
    swp = nkf.vqs.samples_with_probability(vstate)
    b = vstate.is_state(p0, reference=swp)  # in-memory SamplesWithProb

    npz = str(tmp_path / "ref")
    swp.save(npz)
    c = vstate.is_state(p0, reference=npz)  # .npz path

    nkpath = str(tmp_path / "state.nk")
    nkf.vqs.save(nkpath, vstate)
    d = ISState.from_nqxpack(nkpath, p0)  # nqxpack

    chis = [np.asarray(s.expect(obs).mean) for s in (a, b, c, d)]
    esss = [float(s.ess) for s in (a, b, c, d)]
    for chi in chis[1:]:
        np.testing.assert_allclose(chi, chis[0], rtol=1e-6)
    for ess in esss[1:]:
        np.testing.assert_allclose(ess, esss[0], rtol=1e-6)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def test_is_state_susceptibility_result(vstate, obs):
    p0 = vstate.parameter_array[0]
    res = vstate.is_state(p0).expect(obs)
    n_params = p0.size
    assert isinstance(res, ISMatrixResult)
    assert res.mean.shape == (n_params, n_params)
    assert res.error_of_mean.shape == (n_params, n_params)
    assert np.all(np.isfinite(np.asarray(res.mean)))
    assert np.all(np.isfinite(np.asarray(res.error_of_mean)))


def test_is_state_operator_result(vstate):
    p0 = vstate.parameter_array[0]
    ham = make_ising(vstate.hilbert_physical)(p0)
    res = vstate.is_state(p0).expect(ham)
    assert isinstance(res, ISResult)
    assert np.isfinite(float(res.mean.real))
    assert np.isfinite(float(res.error_of_mean))


def test_ess_within_bounds(vstate):
    """At the reference parameters the weights are ~uniform → ess ~ n_samples."""
    p0 = vstate.parameter_array[0]
    st = vstate.is_state(p0)
    st._compute_weights()  # trigger
    ess = float(st.ess)
    assert 0.0 < ess <= st.n_samples + 1e-6
    np.testing.assert_allclose(st.ess_fraction, ess / st.n_samples)
    # Normalized weights sum to 1.
    np.testing.assert_allclose(float(np.asarray(st.normalized_weights).sum()), 1.0)


# ---------------------------------------------------------------------------
# Consistency: IS at the reference point reproduces the direct estimate
# ---------------------------------------------------------------------------


def test_is_at_reference_matches_direct(vstate):
    """When the target equals the reference (psi_tgt == psi_ref), the IS weights
    are uniform (ESS == n) and the IS mean reproduces the plain MC mean — the
    defining consistency property. This pins the target Born exponent: only with
    exponent 2 (matching the machine_pow=2 reference) does log_w vanish here."""
    p0 = vstate.parameter_array[0]
    mc = vstate.get_state(p0, seed=7)
    mc.sample()

    # target_variables == reference variables ⇒ identical target and reference.
    is_st = ISState.from_mc_state(mc, mc.variables)

    n = is_st.n_samples
    np.testing.assert_allclose(
        np.asarray(is_st.normalized_weights), np.full(n, 1.0 / n), rtol=1e-6
    )
    np.testing.assert_allclose(float(is_st.ess), n, rtol=1e-6)

    ham = make_ising(vstate.hilbert_physical)(p0)
    np.testing.assert_allclose(
        complex(is_st.expect(ham).mean), complex(mc.expect(ham).mean), rtol=1e-5
    )


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------


def test_samples_with_prob_npz_roundtrip(vstate, tmp_path):
    swp = nkf.vqs.samples_with_probability(vstate)
    path = str(tmp_path / "ref")
    swp.save(path)
    loaded = SamplesWithProb.load(path)
    np.testing.assert_allclose(np.asarray(loaded.samples), np.asarray(swp.samples))
    np.testing.assert_allclose(np.asarray(loaded.log_probs), np.asarray(swp.log_probs))


def test_samples_with_prob_load_appends_npz(vstate, tmp_path):
    """save writes <path>.npz; load(<path>) must find it."""
    swp = nkf.vqs.samples_with_probability(vstate)
    path = str(tmp_path / "ref")
    swp.save(path)
    assert os.path.exists(path + ".npz")
    SamplesWithProb.load(path)  # no .npz suffix — must still resolve


def test_nqxpack_roundtrip_preserves_reference(vstate, tmp_path):
    nkpath = str(tmp_path / "state.nk")
    nkf.vqs.save(nkpath, vstate)
    data = nkf.vqs.load(nkpath, return_dict=True)
    assert isinstance(data["reference"], SamplesWithProb)
    ref = data["reference"]
    src = nkf.vqs.samples_with_probability(vstate)
    np.testing.assert_allclose(np.asarray(ref.samples), np.asarray(src.samples))


# ---------------------------------------------------------------------------
# Shape assertions
# ---------------------------------------------------------------------------


def test_is_state_rejects_non_2d_samples():
    with pytest.raises(ValueError, match="2D"):
        ISState(np.zeros((4,)), np.zeros(4), lambda v, x: x, {})


def test_is_state_rejects_sample_count_mismatch():
    with pytest.raises(ValueError, match="n_samples"):
        ISState(np.zeros((4, 3)), np.zeros(5), lambda v, x: x, {})
