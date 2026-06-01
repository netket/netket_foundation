# API Reference

Full reference documentation for all public classes and functions in `netket_foundation`.

## Hilbert space

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   netket_foundation.ParameterSpace
```

## Variational state

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   netket_foundation.FoundationalQuantumState
```

## Observables

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   netket_foundation.observable.SusceptibilityObservable
```

## Saving and loading

Save a variational state bundled with its importance-sampling reference (sampled
configurations and reference log-probabilities), so that loading is instantaneous
and the IS reference is exactly reproducible.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   netket_foundation.vqs.save
   netket_foundation.vqs.load
   netket_foundation.vqs.samples_with_probability
```

## Importance sampling

Estimate expectation values of a target state from samples drawn from a reference
distribution, together with the corresponding result and diagnostics types.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   netket_foundation.expectation_value.ISState
   netket_foundation.expectation_value.SamplesWithProb
   netket_foundation.expectation_value.ISResult
   netket_foundation.expectation_value.ISMatrixResult
```

## Driver

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   netket_foundation.VMC_NG
```

## Operators

See the {doc}`operators` page for full documentation of all operator classes and factory functions.

```{toctree}
:hidden:

operators
```
