import argparse
import csv
import os
import statistics
import time

import numba
import numpy as np

from numba_reorder import reorder_index_numba
from reorder import reorder_index_recalculation

COMPARE_SHAPES = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
RUNS = 3
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")


def timed(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def compare_python_numba():
    numba.set_num_threads(1)
    reorder_index_numba(np.zeros(16, np.uint8), np.zeros(16, np.uint8), (4, 4))

    for shape in COMPARE_SHAPES:
        src = np.ones(int(np.prod(shape)), dtype=np.uint8)
        dst = np.zeros_like(src)

        t_py = timed(lambda: reorder_index_recalculation(src, shape))
        t_nb = timed(lambda: reorder_index_numba(src, dst, shape))

        print(
            f"shape={shape} python={t_py:.6f}s ({2 * src.size / t_py / 1e9:.4f} GB/s) "
            f"numba={t_nb:.6f}s ({2 * src.size / t_nb / 1e9:.4f} GB/s) "
            f"speedup={t_py / t_nb:.1f}x"
        )


def sweep(log2n, max_threads, csv_path):
    n = 2**log2n
    src = np.ones(n, dtype=np.uint8)
    dst = np.zeros_like(src)
    traffic_bytes = 2 * n * src.itemsize

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["rank", "shape", "processes", "run1", "run2", "run3",
                  "median_time", "traffic_bytes", "bandwidth_gbs", "speedup"]
        writer.writerow(header)
        print(",".join(header))

        for rank in [r for r in range(1, log2n + 1) if log2n % r == 0]:
            shape = (2 ** (log2n // rank),) * rank
            t1 = None

            for p in range(1, max_threads + 1):
                numba.set_num_threads(p)
                reorder_index_numba(src, dst, shape)  # warm-up
                runs = [timed(lambda: reorder_index_numba(src, dst, shape)) for _ in range(RUNS)]
                median = statistics.median(runs)
                t1 = t1 or median

                row = [rank, "x".join(map(str, shape)), p, *(f"{t:.6f}" for t in runs),
                       f"{median:.6f}", traffic_bytes, f"{traffic_bytes / median / 1e9:.4f}",
                       f"{t1 / median:.3f}"]
                writer.writerow(row)
                f.flush()
                print(",".join(map(str, row)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log2n", type=int, default=32)
    parser.add_argument("--max-threads", type=int, default=numba.config.NUMBA_NUM_THREADS)
    parser.add_argument("--csv", default=os.path.join(RUNS_DIR, "results.csv"))
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare_python_numba()
    else:
        sweep(args.log2n, args.max_threads, args.csv)
