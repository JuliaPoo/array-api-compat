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
    x = numpy.swapaxes(x, axis, -1)
    vals, args = lax.top_k(x, k)
    vals = numpy.swapaxes(vals, axis, -1)
    args = numpy.swapaxes(args, axis, -1)
    return vals, args


__all__ = ['top_k']