# Getting Started

## Installation

Install `netket_foundation` from source:

```bash
uv add git+https://github.com/NeuralQXLab/netket_foundation.git
```

Or in development mode after cloning:

```bash
git clone https://github.com/NeuralQXLab/netket_foundation.git
cd netket_foundation
uv run python
```

See [NetKet's documentation](https://netket.readthedocs.io/en/latest/install.html) for more complete installation instructions and how to handle multi-node and multi-GPU setups.

## Minimal Example

The following trains a foundational neural quantum state over a 1D Ising chain
with varying transverse field strengths $h \in [0.8, 1.2]$:

```python
import jax.numpy as jnp
import netket as nk
import netket_foundation as nkf
import optax

from netket_foundation._src.model.vit import ViTFNQS

# --- Hilbert space + parameter space ---
# The parameter space spans the values of `h` we want to learn jointly.
hi = nk.hilbert.Spin(s=0.5, N=10)
ps = nkf.ParameterSpace(N=1, min=0.8, max=1.2)

# --- Foundation model (Vision-Transformer ansatz) ---
ma = ViTFNQS(
    num_layers=2,
    d_model=12,
    heads=4,
    L_eff=hi.size // 2,  # number of patches = hi.size / b
    n_coups=ps.size,     # number of parameters carried by the parameter space
    b=2,                 # patch size
)

# --- Variational state ---
sa = nk.sampler.MetropolisLocal(hi, n_chains=5016)
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=8, seed=1)
# Place the replicas on a grid over the parameter space.
vs.parameter_array = jnp.linspace(0.8, 1.2, vs.n_replicas).reshape(-1, 1)

# --- Parametrized Hamiltonian ---
# `create_operator` is evaluated for each sampled value of the parameters.
def create_operator(params):
    h = params[0]
    ha_x = sum(nkf.operator.sigmax(hi, i) for i in range(hi.size))
    ha_zz = sum(
        nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + 1) % hi.size)
        for i in range(hi.size)
    )
    return -h * ha_x - ha_zz

ha = nkf.operator.ParametrizedOperator(hi, ps, create_operator)

# --- Optimize ---
optimizer = optax.sgd(0.005)
gs = nkf.VMC_NG(ha, optimizer, variational_state=vs, diag_shift=1e-4)
gs.run(n_iter=300)
```

:::{note}
See the [`examples/`](https://github.com/NeuralQXLab/netket_foundation/tree/main/examples)
directory for full runnable scripts.
:::

## Next Steps

- {doc}`tutorials/index` — detailed walkthroughs
- {doc}`api/index` — full API reference
