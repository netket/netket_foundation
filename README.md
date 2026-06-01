# NetKet Foundation

NetKet Foundation is an extension library built on top of [NetKet](https://github.com/netket/netket) and [JAX](https://github.com/google/jax).
It provides tools to train and evaluate *foundation neural quantum states* over families of Hamiltonians parameterized by couplings or disorder realizations.

The project is designed to reuse NetKet's ecosystem (samplers, operators, logging, drivers) while introducing foundational workflows where one model is optimized across many parameter points at once.

## Main Additions

Compared to base NetKet, this package introduces:

- `ParameterSpace`: a Hilbert space class for Hamiltonian/control parameters.
- `FoundationalQuantumState`: a variational state that samples physical configurations together with parameter replicas.
- `ParametrizedOperator`: operators whose matrix elements are generated from per-sample parameters.
- `VMC_NG`: a natural-gradient VMC driver adapted to foundational training.

## Installation

NetKet Foundation requires Python 3.11+.

Clone the repository and install dependencies with `uv`:

```sh
git clone https://github.com/NeuralQXLab/netket_foundation.git
cd netket_foundation
uv sync
```

Using [uv](https://docs.astral.sh/uv/getting-started/installation/) gives you the exact dependency set used for development, pinned in `uv.lock`.

You can also use pip, but we recommend `uv` for reproducibility:

```sh
pip install -e .
```

## Minimal Usage

### 1) Define a foundational state over a parameter range

```python
import jax.numpy as jnp
import netket as nk
import netket_foundation as nkf

from netket_foundation._src.model.vit import ViTFNQS

hi = nk.hilbert.Spin(0.5, 10)
ps = nkf.ParameterSpace(N=1, min=0.8, max=1.2)

model = ViTFNQS(
	num_layers=2,
	d_model=12,
	heads=4,
	L_eff=hi.size // 2,
	n_coups=ps.size,
	b=2,
	complex=False,
	disorder=False,
	transl_invariant=True,
	two_dimensional=False,
)

sampler = nk.sampler.MetropolisLocal(hi, n_chains=2048)
vstate = nkf.FoundationalQuantumState(sampler, model, ps, n_replicas=8, seed=1)

# Define the coupling used during training
vstate.parameter_array = jnp.linspace(0.8, 1.2, vstate.n_replicas).reshape(-1, 1)
```

### 2) Build a parameter-dependent operator

```python
import netket_foundation as nkf

def create_operator(params):
	h = params[0]
	ha_x = sum(nkf.operator.sigmax(hi, i) for i in range(hi.size))
	ha_zz = sum(
		nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + 1) % hi.size)
		for i in range(hi.size)
	)
	return -h * ha_x - ha_zz

ham = nkf.operator.ParametrizedOperator(hi, ps, create_operator)
```

### 3) Optimize with foundational natural-gradient VMC

```python
import optax
import netket_foundation as nkf

optimizer = optax.sgd(5e-3)
driver = nkf.VMC_NG(ham, optimizer, variational_state=vstate, diag_shift=1e-4)
driver.run(100)
```

## Examples

See complete scripts in:

- `examples/ising1d.py`: foundational training on the Ising chain.
- `examples/ising1d_uniform.py`: foundational training on the disordered Ising chain.
- `examples/susceptibility.py`: full fidelity-susceptibility workflow, including IS.
- `examples/susceptibility_to_precision.py`: minimal example for adaptive susceptibility estimation.
