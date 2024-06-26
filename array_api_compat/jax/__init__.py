import jax
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Tuple

    from ..common._typing import Array


def top_k(
    x: Array,
    k: int,
    /,
    axis: Optional[int] = None,
    *,
    largest: bool = True,
) -> Tuple[Array, Array]:

    # `swapaxes` is used to implement
    # the `axis` kwarg
    x = jax.numpy.swapaxes(x, axis, -1)
    vals, args = jax.lax.top_k(x, k)
    vals = jax.numpy.swapaxes(vals, axis, -1)
    args = jax.numpy.swapaxes(args, axis, -1)
    return vals, args


__all__ = ['top_k']