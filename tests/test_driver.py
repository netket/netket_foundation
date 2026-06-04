"""Integration test for VMC_SR driver: single optimisation step."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import optax
import netket as nk
import netket_foundation as nkf
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
def ham(hi, ps):
    return nkf.operator.ParametrizedOperator(hi, ps, make_ising(hi))


@pytest.fixture(scope="module")
def driver(sampler, model, ps, ham):
    vs = make_vstate(sampler, model, ps, seed=99)
    optimizer = optax.sgd(0.01)
    return nkf.VMC_SR(ham, optimizer, variational_state=vs, diag_shift=1e-3)


def test_single_step_runs(driver):
    """A single optimisation step completes without raising."""
    driver.run(1)


def test_energy_finite(driver):
    """After one step the per-replica energies are all finite real numbers."""
    result = driver.state.expect(driver._ham)
    for stats in result:
        e = float(stats.mean.real)
        assert np.isfinite(e), f"Energy is not finite: {e}"


def test_variables_change_after_step(driver):
    """Parameters are updated (not identical) after at least one gradient step."""
    leaves = jnp.concatenate(
        [jnp.ravel(v) for v in jax.tree_util.tree_leaves(driver.state.variables)]
    )
    assert not jnp.all(leaves == 0)


# --------------------------------------------------------------------------- #
# Parity tests: on_the_fly=True vs on_the_fly=False (and dense NTK vs SR).
#
# Mirrors netket's test/driver/test_vmc_ngd.py::test_SRt_vs_SR: starting from an
# identical seeded state, a few optimisation steps must produce numerically
# identical parameters and losses regardless of which formulation is used.
# --------------------------------------------------------------------------- #

N_ITERS = 5
SEED = 1234


def _make_driver(
    *, complex, use_ntk, on_the_fly, momentum, diag_shift=1e-3, chunk_size_bwd=None
):
    """Fresh driver from an identical seeded state for each configuration."""
    hi = make_hilbert()
    ps = make_parameter_space()
    sampler = make_sampler(hi)
    model = make_model(ps, complex=complex)
    ham = nkf.operator.ParametrizedOperator(hi, ps, make_ising(hi))
    vs = make_vstate(sampler, model, ps, seed=SEED)
    optimizer = optax.sgd(0.01)
    return nkf.VMC_SR(
        ham,
        optimizer,
        variational_state=vs,
        diag_shift=diag_shift,
        momentum=momentum,
        use_ntk=use_ntk,
        on_the_fly=on_the_fly,
        chunk_size_bwd=chunk_size_bwd,
    )


def _run(driver):
    logger = nk.logging.RuntimeLog()
    driver.run(N_ITERS, out=logger)
    return logger


@pytest.mark.parametrize(
    "complex", [pytest.param(False, id="real"), pytest.param(True, id="complex")]
)
@pytest.mark.parametrize(
    "momentum", [pytest.param(None, id=""), pytest.param(0.5, id="momentum")]
)
def test_onthefly_vs_dense_ntk(complex, momentum):
    """on_the_fly=True must match the dense NTK (on_the_fly=False) update."""
    driver_otf = _make_driver(
        complex=complex, use_ntk=True, on_the_fly=True, momentum=momentum
    )
    driver_dense = _make_driver(
        complex=complex, use_ntk=True, on_the_fly=False, momentum=momentum
    )

    log_otf = _run(driver_otf)
    log_dense = _run(driver_dense)

    jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8),
        driver_otf.state.parameters,
        driver_dense.state.parameters,
    )

    if jax.process_index() == 0:
        e_otf = log_otf.data["Mean Energy"]["Mean"]
        e_dense = log_dense.data["Mean Energy"]["Mean"]
        np.testing.assert_allclose(e_otf, e_dense, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    "complex", [pytest.param(False, id="real"), pytest.param(True, id="complex")]
)
def test_onthefly_vs_sr(complex):
    """on_the_fly NTK must match the plain SR (QGT) update, as in netket."""
    driver_otf = _make_driver(
        complex=complex, use_ntk=True, on_the_fly=True, momentum=None
    )
    driver_sr = _make_driver(
        complex=complex, use_ntk=False, on_the_fly=False, momentum=None
    )

    _run(driver_otf)
    _run(driver_sr)

    jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7),
        driver_otf.state.parameters,
        driver_sr.state.parameters,
    )


@pytest.mark.parametrize(
    "complex", [pytest.param(False, id="real"), pytest.param(True, id="complex")]
)
def test_onthefly_chunked_vs_dense(complex):
    """Chunked on_the_fly NTK must match the dense (unchunked) NTK update."""
    driver_otf = _make_driver(
        complex=complex,
        use_ntk=True,
        on_the_fly=True,
        momentum=None,
        chunk_size_bwd=8,
    )
    driver_dense = _make_driver(
        complex=complex, use_ntk=True, on_the_fly=False, momentum=None
    )

    _run(driver_otf)
    _run(driver_dense)

    jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8),
        driver_otf.state.parameters,
        driver_dense.state.parameters,
    )


# # Sharding must be enabled before JAX initialises, so run the sharded parity
# # check in a subprocess with multiple simulated devices. n_chains=4 / n_replicas=4
# # with 2 devices keeps whole replicas on each device (2 | 4).
# _SHARDED_PARITY_SCRIPT = """
# import sys
# sys.path.insert(0, {tests_dir!r})
# import jax, numpy as np, optax
# import netket_foundation as nkf
# from helpers import (
#     make_hilbert, make_parameter_space, make_sampler, make_model,
#     make_ising, make_vstate,
# )
# assert jax.device_count() == 2, jax.device_count()

# def mk(use_ntk, otf):
#     hi = make_hilbert(); ps = make_parameter_space(); sa = make_sampler(hi)
#     m = make_model(ps, complex={complex})
#     ham = nkf.operator.ParametrizedOperator(hi, ps, make_ising(hi))
#     vs = make_vstate(sa, m, ps, seed=1234)
#     return nkf.VMC_SR(ham, optax.sgd(0.01), variational_state=vs,
#                       diag_shift=1e-3, use_ntk=use_ntk, on_the_fly=otf)

# d_otf = mk(True, True); d_dense = mk(True, False)
# d_otf.run(5); d_dense.run(5)
# jax.tree_util.tree_map(
#     lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8),
#     d_otf.state.parameters, d_dense.state.parameters,
# )
# print("SHARDED_PARITY_OK")
# """


# @pytest.mark.parametrize(
#     "complex", [pytest.param(False, id="real"), pytest.param(True, id="complex")]
# )
# def test_onthefly_vs_dense_sharded(complex):
#     """on_the_fly NTK matches the dense NTK update under jax sharding (shard_map)."""
#     import os
#     import subprocess

#     tests_dir = os.path.dirname(os.path.abspath(__file__))
#     script = _SHARDED_PARITY_SCRIPT.format(tests_dir=tests_dir, complex=complex)
#     env = {
#         **os.environ,
#         "NETKET_SHARDING": "1",
#         "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
#     }
#     res = subprocess.run(
#         [sys.executable, "-c", script],
#         env=env,
#         capture_output=True,
#         text=True,
#     )
#     assert "SHARDED_PARITY_OK" in res.stdout, res.stdout + "\n" + res.stderr
