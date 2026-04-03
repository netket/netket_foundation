# Getting Started

## Installation

Install `netket_foundation` from source:

```bash
pip install git+https://github.com/NeuralQXLab/netket_foundation.git
```

Or in development mode after cloning:

```bash
git clone https://github.com/NeuralQXLab/netket_foundation.git
cd netket_foundation
pip install -e .
```

### Requirements

- Python ≥ 3.11
- NetKet ≥ 3.22
- JAX (installed via NetKet)

## Minimal Example

The following trains a foundational neural quantum state over a 1D Ising chain
with varying transverse field strengths $h \in [0.5, 1.5]$:

```python
import netket as nk
import netket_foundation as nkf

# --- Hilbert space + parameter space ---
hi = nk.hilbert.Spin(s=0.5, N=10)
param_space = nkf.ParameterSpace({"h": [0.5, 0.8, 1.0, 1.2, 1.5]})

# --- Hamiltonian factory ---
def make_hamiltonian(h):
    return nk.operator.Ising(hi, nk.graph.Chain(10), h=h)

# --- Foundation model ---
model = nkf.model.ViTFNQS(...)

# --- Variational state ---
vqs = nkf.FoundationalQuantumState(hi, param_space, model, ...)

# --- Optimize ---
gs = nkf.VMC_NG(make_hamiltonian, vqs, ...)
gs.run(n_iter=300)
```

:::{note}
See the [`examples/`](https://github.com/NeuralQXLab/netket_foundation/tree/main/examples)
directory for full runnable scripts.
:::

## Next Steps

- {doc}`tutorials/index` — detailed walkthroughs
- {doc}`api/index` — full API reference
