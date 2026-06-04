from typing import Any, Callable

import jax.numpy as jnp
import flax.linen as nn
from netket.nn.activation import log_cosh

from .fermi_vit import foundation_ViT_trans_equi
from .fermi_backflow import foundation_backflow, foundation_generalized_backflow
from .fermi_jastrow import foundation_fermi_Jastrow_MLP
from .prod_module import ProductModule


def ViTFermionicFNQS(
    hilbert,
    graph,
    n_coups: int,
    num_layers: int,
    d_model: int,
    heads: int,
    b: int = 2,
    d_latent: int | None = None,
    two_dimensional: bool = False,
    complex: bool = False,
    is_equivariant: bool = False,
    make_it_invariant: bool = False,
    generalized: bool = False,
    use_jastrow: bool = True,
    jastrow_activation: Callable = nn.gelu,
    mean_field_init: str = "default",
    enforce_spin_flip: bool = False,
    out_activation: Any = nn.tanh,
    param_dtype: Any = jnp.float64,
) -> nn.Module:
    """Assemble the standard fermionic FNQS ansatz (backflow + optional Jastrow).

    Derives ``d_output`` and ``n_patches`` from ``hilbert`` and ``graph``
    so the caller only needs to specify the architecture hyper-parameters.

    Parameters
    ----------
    hilbert:
        A ``SpinOrbitalFermions`` Hilbert space.
    graph:
        The lattice graph (used for equivariant embedding and Fermi-sea init).
    n_coups:
        Number of coupling parameters in the ``ParameterSpace`` (``ps.size``).
    num_layers:
        Number of transformer encoder layers.
    d_model:
        Model width (used for both the ViT trunk and the Jastrow MLP).
    heads:
        Number of attention heads.
    b:
        Patch size. For 1D: ``graph.n_nodes`` must be divisible by ``b``.
        For 2D: ``graph.n_nodes`` must be divisible by ``b²``.
    d_latent:
        Latent dimension of the ViT output head. Defaults to ``d_model``
        when not set.
    two_dimensional:
        If ``True``, use 2D patch embedding and a 2D equivariant encoder.
        ``n_patches`` is derived as ``graph.n_nodes // b²`` instead of
        ``graph.n_nodes // b``.
    complex:
        If ``True``, the ViT output head produces complex-valued amplitudes.
    is_equivariant:
        If ``True``, use a translation-equivariant ViT encoder.
    make_it_invariant:
        If ``True``, project the ViT output to a translation-invariant scalar.
        Requires ``is_equivariant=True``.
    generalized:
        If ``True``, use ``foundation_generalized_backflow`` which forms a
        single Slater determinant over the mixed spin-orbital basis.
        ``d_output`` is scaled by 2 accordingly. If ``False`` (default),
        use ``foundation_backflow`` with separate UP/DOWN determinants.
    use_jastrow:
        If ``True`` (default) wraps the backflow with a Jastrow MLP via
        ``ProductModule``. Set to ``False`` to use backflow alone.
    jastrow_activation:
        Hidden-layer activation for the Jastrow MLP (default: ``nn.gelu``).
    mean_field_init:
        Mean-field initialisation for the Slater matrix. ``"default"`` uses
        random Lecun-normal; ``"fermi_sea"`` initialises from the tight-binding
        ground state. Ignored when ``generalized=True``.
    enforce_spin_flip:
        Project the ansatz onto the spin-flip symmetric subspace.
    out_activation:
        Activation applied to the ViT output head (default: ``nn.tanh``).
    param_dtype:
        Parameter dtype for all sub-modules (default: ``jnp.float64``).

    Returns
    -------
    nn.Module
        A ``ProductModule(jastrow, backflow)`` when ``use_jastrow=True``,
        or a bare backflow module when ``use_jastrow=False``.
    """
    _d_latent = d_latent if d_latent is not None else d_model
    n_patches = graph.n_nodes // (b**2 if two_dimensional else b)
    d_output = (2 if generalized else 1) * hilbert.n_orbitals * hilbert.n_fermions

    vit = foundation_ViT_trans_equi(
        n_layers=num_layers,
        d_model=d_model,
        d_output=d_output,
        d_latent=_d_latent,
        heads=heads,
        b=b,
        is_2d=two_dimensional,
        n_patches=n_patches,
        n_coups=n_coups,
        graph=graph,
        is_equivariant=is_equivariant,
        make_it_invariant=make_it_invariant,
        complex=complex,
        out_activation=out_activation,
        param_dtype=param_dtype,
    )

    if generalized:
        backflow = foundation_generalized_backflow(
            model=vit,
            hilbert=hilbert,
            enforce_spin_flip=enforce_spin_flip,
            param_dtype=param_dtype,
        )
    else:
        backflow = foundation_backflow(
            model=vit,
            hilbert=hilbert,
            graph=graph,
            mean_field_init=mean_field_init,
            enforce_spin_flip=enforce_spin_flip,
            param_dtype=param_dtype,
        )

    if not use_jastrow:
        return backflow

    jastrow = foundation_fermi_Jastrow_MLP(
        n_layers=num_layers,
        n_coups=n_coups,
        d_model=d_model,
        param_dtype=param_dtype,
        activation=jastrow_activation,
        out_activation=log_cosh,
    )
    return ProductModule(jastrow, backflow)
