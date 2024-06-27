from jax.numpy import * # quick hack
from jax import *


def top_k(
    x,
    k,
    /,
    axis=None,
    *,
    largest=True,
):

    # `swapaxes` is used to implement
    # the `axis` kwarg
    # x = numpy.swapaxes(x, axis, -1)
    # vals, args = lax.top_k(x, k)
    # vals = numpy.swapaxes(vals, axis, -1)
    # args = numpy.swapaxes(args, axis, -1)
    # return vals, args

    # The largest keyword can't be implemented with `jax.lax.top_k`
    # efficiently so am using `jax.numpy` for now
    if k <= 0:
        raise ValueError(f'k(={k}) provided must be positive.')

    positive_axis: int
    _arr = asarray(a)
    if axis is None:
        arr = _arr.ravel()
        positive_axis = 0
    else:
        arr = _arr
        positive_axis = axis if axis > 0 else axis % arr.ndim

    slice_start = (s_[:],) * positive_axis
    if largest:
        indices_array = argpartition(arr, -k, axis=axis)
        slice = slice_start + (s_[-k:],)
        topk_indices = indices_array[slice]
    else:
        indices_array = argpartition(arr, k-1, axis=axis)
        slice = slice_start + (s_[:k],)
        topk_indices = indices_array[slice]

    topk_values = take_along_axis(arr, topk_indices, axis=axis)

    return (topk_values, topk_indices)


__all__ = ['top_k']