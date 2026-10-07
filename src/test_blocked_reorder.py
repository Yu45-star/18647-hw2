import numba
import numpy as np

from blocked_reorder import reorder_blocked_numba, reorder_rowwise_numba
from numba_reorder import reorder_index_numba

BLOCK_SIZES = [1, 8, 16, 32, 64, 128, 256]

TEST_SHAPES = [
    (1, 1),
    (5, 5),        # smaller than every block > 5
    (16, 16),      # divisible by 8 and 16
    (17, 17),      # one-element boundary tiles
    (64, 64),
    (100, 100),
    (256, 256),    # divisible by every block size
    (300, 300),    # partial tiles for every block size > 1
    (33, 65),      # rectangular, partial in both dimensions
    (128, 40),
    (1, 37),
    (37, 1),
]


def get_expected(src, shape):
    # NumPy is used only as a test oracle.
    return src.reshape(shape).T.reshape(-1)


def test_blocked():
    for num_threads in [1, 2, 4]:
        numba.set_num_threads(num_threads)

        for shape in TEST_SHAPES:
            size = shape[0] * shape[1]
            # int64 makes every element unique; uint8 is the benchmarked dtype.
            sources = [
                np.arange(size, dtype=np.int64),
                (np.arange(size) % 251).astype(np.uint8),
            ]

            for src in sources:
                expected = get_expected(src, shape)

                for block in BLOCK_SIZES:
                    dst = np.zeros_like(src)
                    reorder_blocked_numba(src, dst, shape, block)

                    assert np.array_equal(dst, expected), (shape, block, src.dtype)

                dst = reorder_rowwise_numba(src, np.zeros_like(src), shape)
                assert np.array_equal(dst, expected), (shape, "rowwise", src.dtype)

        print(f"Blocked + rowwise: threads={num_threads}: passed")

    print("All blocked tests passed.")


def test_blocked_matches_naive():
    numba.set_num_threads(4)
    rng = np.random.default_rng(0)

    for shape in [(257, 257), (512, 384), (1000, 1000)]:
        src = rng.integers(0, 256, shape[0] * shape[1], dtype=np.uint8)
        naive = reorder_index_numba(src, np.zeros_like(src), shape)

        for block in BLOCK_SIZES:
            blocked = reorder_blocked_numba(src, np.zeros_like(src), shape, block)

            assert np.array_equal(blocked, naive), (shape, block)

    print("Blocked vs. naive Numba: passed")


if __name__ == "__main__":
    test_blocked()
    test_blocked_matches_naive()
