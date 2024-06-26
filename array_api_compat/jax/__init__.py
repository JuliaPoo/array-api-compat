import jax

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
    x = jax.numpy.swapaxes(x, axis, -1)
    vals, args = jax.lax.top_k(x, k)
    vals = jax.numpy.swapaxes(vals, axis, -1)
    args = jax.numpy.swapaxes(args, axis, -1)
    return vals, args


__all__ = ['top_k']