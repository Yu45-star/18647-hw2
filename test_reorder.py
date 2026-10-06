import numba
import numpy as np

from reorder import (
    reorder_index_recalculation,
    reorder_iterative,
    reorder_recursive_1d,
    reorder_recursive_nd,
)

from parallel_reorder import (
    parallel_reorder_shared,
)

from numba_reorder import (
    reorder_index_numba,
)

TEST_SHAPES = [
    (5,),
    (2, 3),
    (2, 3, 4),
    (2, 3, 4, 5),
    (3, 2, 5, 4),
]


def get_expected(src, shape):
    axes = tuple(
        range(
            len(shape) - 1,
            -1,
            -1
        )
    )

    return (
        src
        .reshape(shape)
        .transpose(axes)
        .reshape(-1)
    )


def test_serial():
    for shape in TEST_SHAPES:
        size = int(np.prod(shape))
        src = np.arange(size)

        expected = get_expected(
            src,
            shape
        )

        dst_index = reorder_index_recalculation(
            src,
            shape
        )

        dst_iterative = reorder_iterative(
            src,
            shape
        )

        dst_recursive_1d = reorder_recursive_1d(
            src,
            shape
        )

        dst_recursive_nd = reorder_recursive_nd(
            src,
            shape
        )

        assert np.array_equal(
            dst_index,
            expected
        )

        assert np.array_equal(
            dst_iterative,
            expected
        )

        assert np.array_equal(
            dst_recursive_1d,
            expected
        )

        assert np.array_equal(
            dst_recursive_nd,
            expected
        )

        assert np.array_equal(
            dst_index,
            dst_iterative
        )

        assert np.array_equal(
            dst_index,
            dst_recursive_1d
        )

        assert np.array_equal(
            dst_index,
            dst_recursive_nd
        )

        print(
            f"Serial: shape={shape}: passed"
        )

    print("All serial tests passed.")


def test_parallel_shared():
    for shape in TEST_SHAPES:
        size = int(np.prod(shape))

        src = np.arange(
            size,
            dtype=np.int64
        )

        expected = get_expected(
            src,
            shape
        )

        for num_processes in [1, 2, 4]:
            _, dst = parallel_reorder_shared(
                shape,
                num_processes,
                return_result=True,
                src=src,
            )

            assert np.array_equal(
                dst,
                expected
            )

            print(
                f"Shared: shape={shape}, "
                f"processes={num_processes}: passed"
            )

    print("All shared-memory tests passed.")

def test_numba():
    shapes = TEST_SHAPES + [(7,), (2,) * 6, (4,) * 4]

    for num_threads in [1, 2, 4]:
        numba.set_num_threads(num_threads)

        for shape in shapes:
            size = int(np.prod(shape))
            src = np.arange(size, dtype=np.int64)
            dst = np.empty_like(src)

            reorder_index_numba(src, dst, shape)

            assert np.array_equal(dst, get_expected(src, shape))

        print(f"Numba: threads={num_threads}: passed")

    print("All Numba tests passed.")


if __name__ == "__main__":
    test_serial()
    test_parallel_shared()
    test_numba()