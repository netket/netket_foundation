# Operators

Reference documentation for all operator classes and factory functions in `netket_foundation.operator`.

## Operator classes

The classes below are operators that specifically work with NetKet foundation. 
The {class}`netket_foundation.operator.ParametrizedOperator` is the general operator used with foundation quantum states, while {class}`~netket_foundation.operator.PauliStringsJax` and {class}`~netket_foundation.operator.FermionOperator2ndJax` are largely equivalent to their NetKet counterparts but have a few minor modifications necessary to build those operators within jit contexts. 
We will, over time, upstream those changes to NetKet.
The {class}`~netket_foundation.operator.EmbedOperator` embeds an operator acting on a subspace into a larger tensor-product Hilbert space.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   netket_foundation.operator.ParametrizedOperator
   netket_foundation.operator.PauliStringsJax
   netket_foundation.operator.FermionOperator2nd
   netket_foundation.operator.FermionOperator2ndJax
   netket_foundation.operator.EmbedOperator
```

## Operator factories

Those commands below can be used to compose Hamiltonians that work with Foundation quantum states.
They behave the same as those within NetKet, but return Foundation-specific classes.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   netket_foundation.operator.sigmax
   netket_foundation.operator.sigmay
   netket_foundation.operator.sigmaz
   netket_foundation.operator.create
   netket_foundation.operator.destroy
   netket_foundation.operator.number
```
