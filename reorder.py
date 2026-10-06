import numpy as np

def coords_to_index(coords, shape):
    index = 0
    stride = 1

    for dim in range(len(shape) - 1, -1, -1):
        index += coords[dim] * stride
        stride *= shape[dim]

    return index

def index_to_coords(index, shape):
    coords = [0] * len(shape)

    for dim in range(len(shape) - 1, -1, -1):
        coords[dim] = index % shape[dim]
        index //= shape[dim]

    return tuple(coords)

def reorder_index_recalculation(src, shape):
    total_size = len(src)

    dst = np.empty_like(src)
    new_shape = shape[::-1]

    for old_index in range(total_size):

        coords = index_to_coords(old_index, shape)

        new_coords = coords[::-1]

        new_index = coords_to_index(new_coords, new_shape)

        dst[new_index] = src[old_index]

    return dst

def increment_coords(coords, shape):
    for dim in range(len(shape) - 1, -1, -1):
        coords[dim] += 1

        if coords[dim] < shape[dim]:
            return

        coords[dim] = 0

def reorder_iterative(src, shape):
    total_size = len(src)

    dst = np.empty_like(src)
    new_shape = shape[::-1]

    coords = [0] * len(shape)

    for old_index in range(total_size):
        new_coords = coords[::-1]

        new_index = coords_to_index(
            new_coords,
            new_shape
        )

        dst[new_index] = src[old_index]

        increment_coords(coords, shape)

    return dst

def reorder_recursive_1d(src, shape):
    dst = np.empty_like(src)

    new_shape = shape[::-1]
    coords = [0] * len(shape)

    def recurse(dim):
        if dim == len(shape):
            old_index = coords_to_index(
                coords,
                shape
            )

            new_coords = coords[::-1]

            new_index = coords_to_index(
                new_coords,
                new_shape
            )

            dst[new_index] = src[old_index]

            return

        for i in range(shape[dim]):
            coords[dim] = i
            recurse(dim + 1)

    recurse(0)

    return dst

def reorder_recursive_nd(src, shape):
    src_nd = src.reshape(shape)
    new_shape = shape[::-1]

    dst_nd = np.empty(new_shape, dtype=src.dtype)

    coords = [0] * len(shape)

    def recurse(dim):
        if dim == len(shape):
            old_coords = tuple(coords)
            new_coords = tuple(coords[::-1])

            dst_nd[new_coords] = src_nd[old_coords]
            return

        for i in range(shape[dim]):
            coords[dim] = i
            recurse(dim + 1)

    recurse(0)

    return dst_nd.reshape(-1)