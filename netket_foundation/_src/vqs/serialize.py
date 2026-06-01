import jax
from flax import serialization
import jax.numpy as jnp

# flake8: noqa: E402
from nqxpack._src.lib_v1.custom_types import (
    register_serialization,
    register_automatic_serialization,
)
from nqxpack._src.contextmgr import current_context

# mcstate
from netket import config
from netket.utils import _serialization as serialization_utils

from jax.sharding import NamedSharding, PartitionSpec as P
from netket_foundation._src.vqs.state import FoundationalQuantumState
from netket_foundation._src.hilbert.parameter_space import ParameterSpace


# serialization
def serialize_FoundationalQuantumState(vstate):
    # Necessary for correctly syncronising samples without serialising
    # the samples themselves
    if vstate._samples is not None:
        sampler_state = vstate._sampler_state_previous
    else:
        sampler_state = vstate.sampler_state

    state_dict = {
        "variables": serialization.to_state_dict(
            serialization_utils.remove_prngkeys(vstate.variables)
        ),
        "sampler_state": serialization.to_state_dict(sampler_state),
        "parameter_array": vstate.parameter_array,
        "n_samples": vstate.n_samples,
        "n_discard_per_chain": vstate.n_discard_per_chain,
        "chunk_size": vstate.chunk_size,
        "n_replicas": vstate.n_replicas,
    }

    sampler_states = {}
    for key in getattr(vstate, "sampler_states", {}).keys():
        # Do not double serialize the default sampler
        if key == "default":
            continue

        samples = vstate._samples_distributions.get(key, None)
        if samples is not None:
            sampler_states[key] = vstate._sampler_states_previous[key]
        else:
            sampler_states[key] = vstate.sampler_states[key]

    state_dict["sampler_states"] = serialization.to_state_dict(sampler_states)

    return state_dict


def deserialize_FoundationalQuantumState(vstate, state_dict):
    import copy

    new_vstate = copy.copy(vstate)
    new_vstate.reset()

    assert state_dict["n_replicas"] == vstate.n_replicas

    vars = jax.tree_util.tree_map(
        jnp.asarray,
        serialization.from_state_dict(vstate.variables, state_dict["variables"]),
    )
    vars = serialization_utils.restore_prngkeys(vstate.variables, vars)
    if config.netket_experimental_sharding:
        vars = jax.tree_util.tree_map(
            lambda x, y: jax.lax.with_sharding_constraint(jnp.asarray(y), x.sharding),
            vstate.variables,
            vars,
        )
    new_vstate.variables = vars

    new_vstate.sampler_state = serialization.from_state_dict(
        vstate.sampler_state, state_dict["sampler_state"]
    )
    new_vstate.n_samples = state_dict["n_samples"]
    new_vstate.n_discard_per_chain = state_dict["n_discard_per_chain"]
    new_vstate.chunk_size = state_dict["chunk_size"]
    new_vstate.parameter_array = state_dict["parameter_array"]

    if "sampler_states" in state_dict:
        ss = new_vstate.sampler_state
        for key, value in state_dict["sampler_states"].items():
            new_vstate.sampler_states[key] = serialization.from_state_dict(ss, value)

    return new_vstate


serialization.register_serialization_state(
    FoundationalQuantumState,
    serialize_FoundationalQuantumState,
    deserialize_FoundationalQuantumState,
)


def _replicate(x):
    if isinstance(x, jax.Array) and not x.is_fully_addressable:
        return jax.lax.with_sharding_constraint(
            x, NamedSharding(jax.sharding.get_abstract_mesh(), P())
        )
    return x


# For model states using frameworks that
def _unpack_variables(state_dict, obj):
    if "variables_structure" in obj:
        variables_flat, _ = jax.tree.flatten(state_dict["variables"])
        variables = jax.tree.unflatten(obj["variables_structure"], variables_flat)
        del obj["variables_structure"], variables_flat
    else:
        variables = state_dict["variables"]
    return variables


def serialize_fqs(
    state: FoundationalQuantumState,
) -> dict:
    asset_manager = current_context().asset_manager

    state_dict = serialization.to_state_dict(state)
    state_dict = jax.tree.map(_replicate, state_dict)
    variables_structure = jax.tree.structure(state.variables)
    asset_manager.write_msgpack("state.msgpack", state_dict)

    return {
        "sampler": state._physical_sampler,
        "model": state._model,  # write the bare model
        "parameter_space": state.parameter_space,
        "variables_structure": variables_structure,
        "n_replicas": state.n_replicas,
        "n_samples": state.n_samples,
    }


def deserialize_fqs(
    obj,
) -> FoundationalQuantumState:
    asset_manager = current_context().asset_manager

    state_dict = asset_manager.read_msgpack("state.msgpack")
    # Todo support this in the init method.
    variables = _unpack_variables(state_dict, obj)  # noqa: F841
    state = FoundationalQuantumState(**obj)  # , variables=variables)
    state = serialization.from_state_dict(state, state_dict)
    return state


register_serialization(FoundationalQuantumState, serialize_fqs, deserialize_fqs)


register_automatic_serialization(
    ParameterSpace,
    N="size",
    min="_min",
    max="_max",
)
