import argparse
import csv
import os
import platform
import statistics
import time

import numba
import numpy as np

from blocked_reorder import reorder_blocked_numba, reorder_rowwise_numba
from numba_reorder import reorder_index_numba

SIZES = [4096, 8192, 16384]
BLOCK_SIZES = [8, 16, 32, 64, 128, 256]
RUNS = 3


def timed(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def measure(fn):
    fn()  # warm-up: JIT compilation and first-touch page faults on dst
    runs = [timed(fn) for _ in range(RUNS)]
    return runs, statistics.median(runs)


def sweep(sizes, blocks, threads, csv_path, txt_path):
    for path in (csv_path, txt_path):
        if os.path.exists(path):
            raise SystemExit(f"{path} already exists; refusing to overwrite it")

    numba.set_num_threads(threads)

    with open(csv_path, "w", newline="") as f, open(txt_path, "w") as log:
        def emit(line):
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        emit(f"# Part 3 blocking sweep  {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        emit(f"# host={platform.node()} machine={platform.machine()} "
             f"numba={numba.__version__} numpy={np.__version__} "
             f"threads={threads} runs={RUNS} dtype=uint8")

        writer = csv.writer(f)
        header = ["matrix_size", "method", "block_size", "threads", "run1", "run2", "run3",
                  "median_time", "traffic_bytes", "bandwidth_gbs", "speedup_vs_naive"]
        writer.writerow(header)
        emit(",".join(header))

        for n in sizes:
            shape = (n, n)
            src = np.ones(n * n, dtype=np.uint8)
            dst = np.zeros_like(src)
            traffic_bytes = 2 * src.size * src.itemsize

            configs = [("naive", "", lambda: reorder_index_numba(src, dst, shape)),
                       ("rowwise", "", lambda: reorder_rowwise_numba(src, dst, shape))]
            configs += [("blocked", b, lambda b=b: reorder_blocked_numba(src, dst, shape, b))
                        for b in blocks]

            t_naive = None
            for method, block, fn in configs:
                runs, median = measure(fn)
                t_naive = t_naive or median

                row = [f"{n}x{n}", method, block, threads, *(f"{t:.6f}" for t in runs),
                       f"{median:.6f}", traffic_bytes, f"{traffic_bytes / median / 1e9:.4f}",
                       f"{t_naive / median:.3f}"]
                writer.writerow(row)
                f.flush()
                emit(",".join(map(str, row)))

            del src, dst


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument("--blocks", type=int, nargs="+", default=BLOCK_SIZES)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--csv", default="../runs/part3_blocking.csv")
    parser.add_argument("--txt", default="../runs/part3_blocking.txt")
    args = parser.parse_args()

    sweep(args.sizes, args.blocks, args.threads, args.csv, args.txt)
