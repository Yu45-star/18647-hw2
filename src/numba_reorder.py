import numpy as np
from numba import njit, prange


# Index recalculation, fused: decode the source linear index digit by digit
# and re-encode it directly with the destination strides (no temporaries).
# In the reversed layout, source dimension k has destination stride prod(shape[:k]).
@njit
def _new_index(old_index, shape, dst_strides):
    # prange yields an unsigned index; mixing it with int64 would promote to float.
    rem = np.int64(old_index)
    new_index = 0

    for k in range(shape.shape[0] - 1, -1, -1):
        new_index += (rem % shape[k]) * dst_strides[k]
        rem //= shape[k]

    return new_index


@njit(parallel=True)
def _reorder_kernel(src, dst, shape, dst_strides):
    for old_index in prange(src.shape[0]):
        dst[_new_index(old_index, shape, dst_strides)] = src[old_index]


def reorder_index_numba(src, dst, shape):
    shape = np.asarray(shape, dtype=np.int64)
    dst_strides = np.concatenate(([1], np.cumprod(shape[:-1]))).astype(np.int64)

    _reorder_kernel(src, dst, shape, dst_strides)

    return dst
