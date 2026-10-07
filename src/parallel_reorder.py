import time
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

from reorder import (
    index_to_coords,
    coords_to_index,
)


def _worker_warmup():
    return None


def reorder_chunk_shared(
    src_shm_name,
    dst_shm_name,
    shape,
    dtype_str,
    total_size,
    start,
    end,
):
    src_shm = shared_memory.SharedMemory(
        name=src_shm_name
    )

    dst_shm = shared_memory.SharedMemory(
        name=dst_shm_name
    )

    try:
        dtype = np.dtype(dtype_str)

        src = np.ndarray(
            (total_size,),
            dtype=dtype,
            buffer=src_shm.buf,
        )

        dst = np.ndarray(
            (total_size,),
            dtype=dtype,
            buffer=dst_shm.buf,
        )

        new_shape = shape[::-1]

        for old_index in range(start, end):
            coords = index_to_coords(
                old_index,
                shape
            )

            new_coords = coords[::-1]

            new_index = coords_to_index(
                new_coords,
                new_shape
            )

            dst[new_index] = src[old_index]

    finally:
        src_shm.close()
        dst_shm.close()


def parallel_reorder_shared(
    shape,
    num_processes,
    dtype=np.uint8,
    return_result=False,
    src=None,
):
    total_size = int(np.prod(shape))

    # For correctness tests, use the caller's data and dtype.
    if src is not None:
        dtype = src.dtype

    dtype = np.dtype(dtype)

    total_bytes = total_size * dtype.itemsize

    src_shm = shared_memory.SharedMemory(
        create=True,
        size=total_bytes,
    )

    dst_shm = shared_memory.SharedMemory(
        create=True,
        size=total_bytes,
    )

    try:
        src_shared = np.ndarray(
            (total_size,),
            dtype=dtype,
            buffer=src_shm.buf,
        )

        dst_shared = np.ndarray(
            (total_size,),
            dtype=dtype,
            buffer=dst_shm.buf,
        )

        # Initialize outside the timed region.
        # The actual values do not matter for performance.
        if src is not None:
            src_shared[:] = np.asarray(src).ravel()
        else:
            src_shared.fill(1)

        dst_shared.fill(0)

        chunk_size = (
            total_size + num_processes - 1
        ) // num_processes

        ranges = []

        for process_id in range(num_processes):
            start = process_id * chunk_size

            end = min(
                start + chunk_size,
                total_size,
            )

            if start >= total_size:
                break

            ranges.append((start, end))

        with ProcessPoolExecutor(
            max_workers=num_processes
        ) as executor:

            # Start worker processes before timing.
            warmups = [
                executor.submit(_worker_warmup)
                for _ in range(num_processes)
            ]

            for task in warmups:
                task.result()

            start_time = time.perf_counter()

            tasks = []

            for start, end in ranges:
                task = executor.submit(
                    reorder_chunk_shared,
                    src_shm.name,
                    dst_shm.name,
                    shape,
                    dtype.str,
                    total_size,
                    start,
                    end,
                )

                tasks.append(task)

            for task in tasks:
                task.result()

            end_time = time.perf_counter()

        elapsed = end_time - start_time

        result = None

        if return_result:
            # Only for small correctness tests.
            # This copy is NOT part of elapsed time.
            result = dst_shared.copy()

        return elapsed, result

    finally:
        src_shm.close()
        src_shm.unlink()

        dst_shm.close()
        dst_shm.unlink()

def worker_warmup():
    return None


def benchmark_reorder_shared(
    shape,
    num_processes,
    dtype=np.uint8,
):
    total_size = int(np.prod(shape))
    dtype = np.dtype(dtype)

    total_bytes = total_size * dtype.itemsize

    # Allocate source and destination directly
    # in shared memory.
    src_shm = shared_memory.SharedMemory(
        create=True,
        size=total_bytes,
    )

    dst_shm = shared_memory.SharedMemory(
        create=True,
        size=total_bytes,
    )

    try:
        src_shared = np.ndarray(
            (total_size,),
            dtype=dtype,
            buffer=src_shm.buf,
        )

        dst_shared = np.ndarray(
            (total_size,),
            dtype=dtype,
            buffer=dst_shm.buf,
        )

        # Initialization is deliberately outside
        # the measured region.
        src_shared.fill(1)
        dst_shared.fill(0)

        chunk_size = (
            total_size + num_processes - 1
        ) // num_processes

        ranges = []

        for process_id in range(num_processes):
            start = process_id * chunk_size

            end = min(
                start + chunk_size,
                total_size,
            )

            if start >= total_size:
                break

            ranges.append((start, end))

        with ProcessPoolExecutor(
            max_workers=num_processes
        ) as executor:

            # Force worker processes to start
            # before measurement.
            warmup_tasks = [
                executor.submit(worker_warmup)
                for _ in range(num_processes)
            ]

            for task in warmup_tasks:
                task.result()

            # ---------------------------
            # Timed region starts here
            # ---------------------------
            start_time = time.perf_counter()

            tasks = []

            for start, end in ranges:
                task = executor.submit(
                    reorder_chunk_shared,
                    src_shm.name,
                    dst_shm.name,
                    shape,
                    dtype.str,
                    total_size,
                    start,
                    end,
                )

                tasks.append(task)

            for task in tasks:
                task.result()

            end_time = time.perf_counter()
            # ---------------------------
            # Timed region ends here
            # ---------------------------

        elapsed = end_time - start_time

        return elapsed

    finally:
        src_shm.close()
        src_shm.unlink()

        dst_shm.close()
        dst_shm.unlink()