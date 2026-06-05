import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

from netket.utils.types import DType

from netket.operator import AbstractOperator


@register_pytree_node_class
class ParametrizedOperator(AbstractOperator):
    """Operator wrapper that evaluates a physical operator from parameters carried in the sample."""

    def __init__(self, hilbert, parameter_space, function):
        super().__init__(hilbert * parameter_space)
        # self._parameter_space = parameter_space
        # self._extended_hilbert = hilbert * parameter_space
        self._function = function

    @property
    def function(self):
        return self._function

    @property
    def dtype(self) -> DType:
        return jnp.float64 if jax.enable_x64 else jnp.float32

    def tree_flatten(self):
        # self._setup()
        data = ()
        metadata = {
            "hilbert": self.hilbert,
            # "parameter_space": self._parameter_space,
            # "extended_hilbert": self._extended_hilbert,
            "function": self._function,
        }
        return data, metadata

    @classmethod
    def tree_unflatten(cls, metadata, data):
        # () = data
        hi = metadata["hilbert"]
        # parameter_space = metadata["parameter_space"]
        # extended_hilbert = metadata["extended_hilbert"]
        function = metadata["function"]
        hilbert = hi.subspaces[0]
        parameter_space = hi.subspaces[1]

        op = cls(hilbert, parameter_space, function=function)
        return op

    @jax.jit
    def get_conn_padded(self, x):
        xr = x.reshape(-1, x.shape[-1])

        xr_physical = xr[:, : self.hilbert.subspaces[0].size]
        xr_params = xr[:, self.hilbert.subspaces[0].size :]

        def _get_conn_pad(xr_params, xr_physical):
            ha = self.function(xr_params)
            return ha.get_conn_padded(xr_physical)

        xs, mels = jax.vmap(_get_conn_pad, in_axes=(0, 0))(xr_params, xr_physical)
        # Reshape y to (C, B, 1, N1), then broadcast to (C, B, M, N1)
        xr_params = jnp.broadcast_to(
            xr_params[:, None, :], xs.shape[:-1] + (xr_params.shape[-1],)
        )
        # Now concatenate along the last axis
        xs = jnp.concatenate([xs, xr_params], axis=-1)

        xs = xs.reshape(*x.shape[:-1], xs.shape[-2], x.shape[-1])
        mels = mels.reshape(*x.shape[:-1], mels.shape[-1])
        return xs, mels

    def with_params(self, params):
        if params.ndim == 1:
            return self.function(params)
        else:
            raise NotImplementedError
