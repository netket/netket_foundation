"""Tests for FoundationalQuantumState."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import netket_foundation as nkf
from netket.stats import Stats, LocalEstimators, LocalEstimatorsBatch
from netket_foundation._src.jax import foundational_log_jacobian
from netket_foundation._src.vqs.fidelity_susceptibility import _combine
from helpers import (
    make_hilbert,
    make_parameter_space,
    make_sampler,
    make_model,
    make_ising,
    make_vstate,
)


@pytest.fixture(scope="module")
def hi():
    return make_hilbert()


@pytest.fixture(scope="module")
def ps():
    return make_parameter_space()


@pytest.fixture(scope="module")
def sampler(hi):
    return make_sampler(hi)


@pytest.fixture(scope="module")
def model(ps):
    return make_model(ps)


@pytest.fixture(scope="module")
def create_ising(hi):
    return make_ising(hi)


@pytest.fixture(scope="module")
def ham(hi, ps, create_ising):
    return nkf.operator.ParametrizedOperator(hi, ps, create_ising)


@pytest.fixture(scope="module")
def vstate(sampler, model, ps):
    return make_vstate(sampler, model, ps)


def test_n_replicas(vstate):
    assert vstate.n_replicas == 4


def test_hilbert_physical(vstate, hi):
    assert vstate.hilbert_physical == hi


def test_parameter_array_shape(vstate, ps):
    pa = vstate.parameter_array
    assert pa.shape == (vstate.n_replicas, ps.size)


def test_parameter_array_assignment(vstate, ps):
    new_pa = jnp.linspace(0.9, 1.1, vstate.n_replicas).reshape(-1, ps.size)
    vstate.parameter_array = new_pa
    np.testing.assert_allclose(vstate.parameter_array, new_pa)
    # restore original
    vstate.parameter_array = jnp.linspace(0.8, 1.2, vstate.n_replicas).reshape(
        -1, ps.size
    )


def test_n_samples(vstate):
    assert vstate.n_samples == 16


def test_samples_shape(vstate, hi, ps):
    vstate.reset()
    samples = vstate.samples
    # shape: (n_chains, chain_length, hi.size + ps.size)
    assert samples.ndim == 3
    assert samples.shape[-1] == hi.size + ps.size


def test_expect_parametrized_operator(vstate, ham):
    """expect on a ParametrizedOperator returns a list of Stats, one per replica."""
    result = vstate.expect(ham)
    assert isinstance(result, list)
    assert len(result) == vstate.n_replicas
    for stats in result:
        assert isinstance(stats, Stats)
        assert np.isfinite(float(stats.mean.real))


def test_local_estimators_returns_container(vstate, ham):
    le = vstate.local_estimators(ham)
    assert isinstance(le, LocalEstimators)
    assert le.data.shape == vstate.samples.shape[:-1]


def test_get_state_returns_mcstate(vstate):
    """get_state extracts a single-parameter MCState."""
    params = vstate.parameter_array[0]
    sub = vstate.get_state(params)
    assert sub.hilbert == vstate.hilbert_physical


def test_is_state_expect_parametrized_operator(vstate, ham, create_ising):
    """ISState evaluates ParametrizedOperator at the target parameters."""
    ref_params = vstate.parameter_array[0]
    target_params = vstate.parameter_array[1]
    mc = vstate.get_state(ref_params, seed=123)
    mc.sample()

    is_state = nkf.expectation_value.ISState.from_mc_state(mc, target_params)

    result_parametrized = is_state.expect(ham)
    result_physical = is_state.expect(create_ising(target_params))

    assert isinstance(result_parametrized, nkf.expectation_value.ISResult)
    np.testing.assert_allclose(result_parametrized.mean, result_physical.mean)
    np.testing.assert_allclose(
        result_parametrized.error_of_mean, result_physical.error_of_mean
    )


def test_is_state_expect_parametrized_operator_requires_single_target(vstate, ham):
    params = vstate.parameter_array[0]
    mc = vstate.get_state(params, seed=123)
    mc.sample()

    is_state = nkf.expectation_value.ISState.from_mc_state(
        mc, vstate.parameter_array[:2]
    )

    with pytest.raises(ValueError, match="single target parameter vector"):
        is_state.expect(ham)


def test_susceptibility_returns_matrix_stats(vstate):
    params = vstate.parameter_array[0]
    mc = vstate.get_state(params, seed=123)
    mc.sample()
    chi = mc.expect(nkf.observable.SusceptibilityObservable(mc.hilbert))

    samples = mc.samples
    if samples.ndim >= 3:
        samples = jax.lax.collapse(samples, 0, 2)
    dlog = foundational_log_jacobian(
        mc.model.apply, mc.variables, samples, mc.chunk_size
    )
    centered = dlog - jnp.mean(dlog, axis=0, keepdims=True)
    chi_ref = jnp.real(
        jnp.mean(jnp.conj(centered[:, :, None]) * centered[:, None, :], axis=0)
    )

    assert chi.mean.shape == (params.size, params.size)
    assert np.isfinite(float(chi.mean[0, 0].real))
    assert np.isfinite(float(chi.error_of_mean[0, 0]))
    np.testing.assert_allclose(chi.mean, chi_ref)


def test_susceptibility_complex_qgt_is_real_hermitian_covariance():
    """Complex log-derivatives need the Hermitian covariance convention."""
    dlog = jnp.asarray(
        [
            [1.0 + 2.0j, 0.5 - 1.0j],
            [-2.0 + 0.25j, 1.5 + 0.5j],
            [0.25 - 0.75j, -1.0 + 2.0j],
        ]
    )
    mean_dlog = jnp.mean(dlog, axis=0)
    mean_outer = jnp.mean(jnp.conj(dlog[:, :, None]) * dlog[:, None, :], axis=0)
    channels_mean = jnp.concatenate([mean_dlog, mean_outer.reshape(-1)])

    expected_qgt = mean_outer - jnp.conj(mean_dlog)[:, None] * mean_dlog[None, :]
    chi = _combine(dlog.shape[-1], channels_mean)

    np.testing.assert_allclose(chi, jnp.real(expected_qgt), atol=1e-7)
    np.testing.assert_allclose(chi, chi.T, atol=1e-7)

    wrong_without_conjugation = jnp.mean(
        (dlog - mean_dlog)[..., :, None] * (dlog - mean_dlog)[..., None, :],
        axis=0,
    )
    assert not np.allclose(np.asarray(chi), np.asarray(wrong_without_conjugation))


def test_susceptibility_observable_local_estimators(vstate):
    params = vstate.parameter_array[0]
    mc = vstate.get_state(params, seed=123)
    mc.sample()

    obs = nkf.observable.SusceptibilityObservable(mc.hilbert)
    le = mc.local_estimators(obs)
    assert isinstance(le, LocalEstimatorsBatch)
    assert le.to_stats().mean.shape == (params.size, params.size)


def test_susceptibility_to_precision_runs(vstate):
    params = vstate.parameter_array[0]
    mc = vstate.get_state(params, seed=123)
    obs = nkf.observable.SusceptibilityObservable(mc.hilbert)
    chi = mc.expect_to_precision(
        obs,
        atol=0.2,
        max_iter=3,
        max_lag=8,
        verbose=True,
    ).get_stats()
    assert chi.mean.shape == (params.size, params.size)
    assert np.isfinite(float(chi.error_of_mean[0, 0]))


def test_unsupported_sampler_raises():
    """Passing a non-Metropolis sampler raises NotImplementedError."""
    hi = make_hilbert()
    ps = make_parameter_space()
    model = make_model(ps)
    exact_sampler = nk.sampler.ExactSampler(hi)
    with pytest.raises(NotImplementedError, match="not supported"):
        nkf.FoundationalQuantumState(exact_sampler, model, ps, n_replicas=4)


def test_n_chains_not_divisible_by_n_replicas_raises():
    """n_replicas that does not divide n_chains raises ValueError."""
    hi = make_hilbert()
    ps = make_parameter_space()
    model = make_model(ps)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=4)
    with pytest.raises(ValueError, match="n_replicas"):
        nkf.FoundationalQuantumState(sampler, model, ps, n_replicas=3)
