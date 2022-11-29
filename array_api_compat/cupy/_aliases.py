from __future__ import annotations

from functools import partial

from ..common import _aliases

from .._internal import get_xp

asarray = asarray_numpy = partial(_aliases._asarray, namespace='numpy')
asarray.__doc__ = _aliases._asarray.__doc__
del partial

import cupy as cp
bool = cp.bool_

acos = get_xp(cp)(_aliases.acos)
acosh = get_xp(cp)(_aliases.acosh)
asin = get_xp(cp)(_aliases.asin)
asinh = get_xp(cp)(_aliases.asinh)
atan = get_xp(cp)(_aliases.atan)
atan2 = get_xp(cp)(_aliases.atan2)
atanh = get_xp(cp)(_aliases.atanh)
bitwise_left_shift = get_xp(cp)(_aliases.bitwise_left_shift)
bitwise_invert = get_xp(cp)(_aliases.bitwise_invert)
bitwise_right_shift = get_xp(cp)(_aliases.bitwise_right_shift)
concat = get_xp(cp)(_aliases.concat)
pow = get_xp(cp)(_aliases.pow)
arange = get_xp(cp)(_aliases.arange)
empty = get_xp(cp)(_aliases.empty)
empty_like = get_xp(cp)(_aliases.empty_like)
eye = get_xp(cp)(_aliases.eye)
full = get_xp(cp)(_aliases.full)
full_like = get_xp(cp)(_aliases.full_like)
linspace = get_xp(cp)(_aliases.linspace)
ones = get_xp(cp)(_aliases.ones)
ones_like = get_xp(cp)(_aliases.ones_like)
zeros = get_xp(cp)(_aliases.zeros)
zeros_like = get_xp(cp)(_aliases.zeros_like)
UniqueAllResult = get_xp(cp)(_aliases.UniqueAllResult)
UniqueCountsResult = get_xp(cp)(_aliases.UniqueCountsResult)
UniqueInverseResult = get_xp(cp)(_aliases.UniqueInverseResult)
unique_all = get_xp(cp)(_aliases.unique_all)
unique_counts = get_xp(cp)(_aliases.unique_counts)
unique_inverse = get_xp(cp)(_aliases.unique_inverse)
unique_values = get_xp(cp)(_aliases.unique_values)
astype = _aliases.astype
std = get_xp(cp)(_aliases.std)
var = get_xp(cp)(_aliases.var)
permute_dims = get_xp(cp)(_aliases.permute_dims)
reshape = get_xp(cp)(_aliases.reshape)
argsort = get_xp(cp)(_aliases.argsort)
sort = get_xp(cp)(_aliases.sort)
sum = get_xp(cp)(_aliases.sum)
prod = get_xp(cp)(_aliases.prod)
ceil = get_xp(cp)(_aliases.ceil)
floor = get_xp(cp)(_aliases.floor)
trunc = get_xp(cp)(_aliases.trunc)

__all__ = _aliases.__all__ + ['asarray', 'asarray_numpy', 'bool', 'arange',
                              'empty', 'empty_like', 'eye', 'full', 'full_like',
                              'linspace', 'ones', 'ones_like', 'zeros', 'zeros_like']
