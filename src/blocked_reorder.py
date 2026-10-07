import numpy as np
from numba import njit, prange


# Part 3: blocked (tiled) rank-2 reorder, i.e. a matrix transpose.
# The source is a flat row-major (rows x cols) array; in the reversed layout
# element (r, c) moves from src[r * cols + c] to dst[c * rows + r].
# The iteration space is split into B x B tiles so that the B source rows and
# B destination rows touched by one tile stay in cache while the tile is processed.
@njit(parallel=True)
def _blocked_kernel(src, dst, rows, cols, block):
    n_tile_rows = (rows + block - 1) // block
    n_tile_cols = (cols + block - 1) // block

    # Static prange scheduling hands each thread a contiguous run of tiles.
    for tile in prange(n_tile_rows * n_tile_cols):
        # prange yields an unsigned index; mixing it with int64 would promote to float.
        t = np.int64(tile)
        r0 = (t // n_tile_cols) * block
        c0 = (t % n_tile_cols) * block
        # Boundary tiles are clipped to the matrix edge.
        r1 = min(r0 + block, rows)
        c1 = min(c0 + block, cols)

        for r in range(r0, r1):
            for c in range(c0, c1):
                dst[c * rows + r] = src[r * cols + c]


def reorder_blocked_numba(src, dst, shape, block):
    rows, cols = shape

    _blocked_kernel(src, dst, rows, cols, block)

    return dst


# Untiled control: the same two-loop transpose without tiling (one full source
# row at a time). It has the same cheap index arithmetic as the blocked kernel,
# so blocked vs. rowwise isolates the locality effect from the div/mod cost of
# the Part 2 index-recalculation kernel.
@njit(parallel=True)
def _rowwise_kernel(src, dst, rows, cols):
    for row in prange(rows):
        r = np.int64(row)
        for c in range(cols):
            dst[c * rows + r] = src[r * cols + c]


def reorder_rowwise_numba(src, dst, shape):
    rows, cols = shape

    _rowwise_kernel(src, dst, rows, cols)

    return dst
