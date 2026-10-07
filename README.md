# 18-647 HW2 — Rank-N Tensor Reordering

This repository contains the implementation and performance study for HW2.

The task is to reorder an arbitrary-rank tensor from C-order layout by reversing the order of its dimensions. For a tensor with shape

```text
(d0, d1, ..., dN-1)
```

the reordered tensor has shape

```text
(dN-1, ..., d1, d0)
```

and coordinates are mapped as

```text
(i0, i1, ..., iN-1)
    ->
(iN-1, ..., i1, i0)
```

The implementation does not use NumPy transpose/permutation routines for the actual tensor reordering. NumPy transpose is used only as a correctness oracle in the test code.

---

## Repository Structure

```text
.
├── src/
│   ├── reorder.py
│   ├── parallel_reorder.py
│   ├── numba_reorder.py
│   ├── blocked_reorder.py
│   ├── test_reorder.py
│   ├── test_blocked_reorder.py
│   ├── benchmark.py
│   └── benchmark_blocking.py
│
├── runs/
│   ├── pilot_2p24.csv
│   ├── pilot_2p24.txt
│   ├── pilot_2p28.csv
│   ├── pilot_2p28.txt
│   ├── final_2p32.csv
│   ├── final_2p32.txt
│   ├── part3_blocking.csv
│   └── part3_blocking.txt
│
├── ai/
├── machine_info.txt
├── HW2_final_results_and_plots.xlsx
├── HW2_part2_analysis.md
├── HW2_part3_analysis.md
└── README.md
```

### Main source files

- `src/reorder.py`  
  Serial implementations:
  - direct index recalculation;
  - iterative coordinate traversal;
  - recursive traversal using a 1-D representation;
  - recursive traversal using an N-D representation.

- `src/parallel_reorder.py`  
  Shared-memory multiprocessing implementation used during development and correctness testing.

- `src/numba_reorder.py`  
  Optimized parallel index-recalculation implementation using Numba and `prange`. This is the implementation used for the main Part 2 performance measurements.

- `src/blocked_reorder.py`  
  Part 3 blocked rank-2 transpose implementation and row-wise control implementation.

- `src/test_reorder.py`  
  Correctness tests for the Part 1 / Part 2 implementations (serial, shared-memory multiprocessing, and Numba).

- `src/test_blocked_reorder.py`  
  Correctness tests for the Part 3 blocked implementation, including boundary tiles and rectangular inputs.

- `src/benchmark.py`  
  Main Part 2 benchmark.

- `src/benchmark_blocking.py`  
  Part 3 blocking/locality benchmark.

---

## Dependencies

The project was developed with Python, NumPy, and Numba.

Install the required packages with:

```bash
pip install numpy numba
```

The final AWS experiments used:

```text
Python 3.9.25
Numba 0.60.0
NumPy 2.0.2
```

---

## Running the Code

Run the scripts from the `src/` directory:

```bash
cd src
```

### Correctness Tests

Run the Part 1 / Part 2 tests:

```bash
python test_reorder.py
```

Run the Part 3 blocked-reordering tests:

```bash
python test_blocked_reorder.py
```

NumPy transpose is used only in the tests to generate the expected result.

---

## Part 2 — Parallel Tensor Reordering

The final implementation uses Numba shared-memory multithreading:

```text
@njit(parallel=True)
prange(...)
```

The number of threads is controlled using Numba's thread-count interface.

### Small Python vs. Numba Comparison

```bash
python benchmark.py --compare
```

### Main Benchmark

The benchmark creates `2^log2n` `uint8` elements and tests all tensor ranks for which an equal-sized dimension can be formed.

For example, a small run:

```bash
python benchmark.py \
    --log2n 24 \
    --max-threads 8 \
    --csv ../runs/results_2p24.csv
```

`benchmark.py` overwrites its CSV output without asking, so use a new file name rather than one of the committed result files.

The final experiment used:

```bash
python benchmark.py \
    --log2n 32 \
    --max-threads 8 \
    --csv ../runs/final_2p32.csv | tee ../runs/final_2p32.txt
```

The script writes the CSV file itself. The `.txt` file is a copy of its console output.

pilot_2p24.* and pilot_2p28.* were preliminary Numba multithreaded runs at smaller problem sizes used to validate scaling and estimate the cost of the final experiment. The final required 4 GiB results are in final_2p32.*.

Each configuration is run three times, and the median runtime is reported.

For the final experiment,

```text
number of elements = 2^32
element type       = uint8
source size        = 4 GiB
destination size   = 4 GiB
```

The effective memory traffic is modeled as one read and one write per element:

```text
traffic_bytes = 2 * number_of_elements
```

and effective bandwidth is calculated as

```text
bandwidth = traffic_bytes / median_runtime
```

---

## Experimental Machine

The final benchmark was run on an AWS machine with:

```text
CPU: Intel Xeon Platinum 8488C
Physical cores: 8
Logical CPUs: 16
Threads per core: 2
Sockets: 1
NUMA nodes: 1
Memory: 30 GiB
```

The main experiment uses 1 through 8 threads so that the reported scaling corresponds to the 8 physical CPU cores.

Full machine information is stored in:

```text
machine_info.txt
```

---

## Part 2 Results

The valid tensor ranks for `2^32` elements are:

```text
1, 2, 4, 8, 16, 32
```

The corresponding shapes are:

```text
Rank 1:  (2^32)
Rank 2:  (65536, 65536)
Rank 4:  (256, 256, 256, 256)
Rank 8:  (16, ..., 16)
Rank 16: (4, ..., 4)
Rank 32: (2, ..., 2)
```

At 8 threads:

| Rank | Effective Bandwidth (GB/s) | Speedup vs. 1 Thread |
|---:|---:|---:|
| 1 | 4.3301 | 7.811x |
| 2 | 0.3639 | 7.236x |
| 4 | 0.3873 | 7.415x |
| 8 | 0.4157 | 7.486x |
| 16 | 0.2005 | 7.811x |
| 32 | 0.0892 | 7.988x |

The implementation shows strong multicore scaling. Higher-rank tensors require substantially more index arithmetic per element, so their effective bandwidth is lower even though their parallel speedup remains close to linear.

Detailed Part 2 results and discussion are in:

```text
HW2_part2_analysis.md
runs/final_2p32.csv
runs/final_2p32.txt
```

---

## Part 3 — Blocking and Locality

For Part 3, I explored blocking/tiling as a locality optimization.

The rank-2 reorder is equivalent to a matrix transpose. The direct implementation reads the source regularly but performs highly strided writes to the destination. The blocked implementation divides the matrix into `B x B` tiles and processes one tile at a time.

A row-wise untiled implementation is also included as a control so that the benefit from cheaper index arithmetic can be separated from the benefit of improved locality.

Run the Part 3 benchmark with:

```bash
python benchmark_blocking.py --threads 8
```

This writes `runs/part3_blocking.csv` and `runs/part3_blocking.txt`. The script refuses to overwrite existing output files, so to rerun it next to the committed results, pass new paths:

```bash
python benchmark_blocking.py --threads 8 \
    --csv ../runs/part3_rerun.csv \
    --txt ../runs/part3_rerun.txt
```

The sizes and block sizes can be changed with `--sizes` and `--blocks`.

The experiment tests:

```text
Matrix sizes:
4096 x 4096
8192 x 8192
16384 x 16384

Block sizes:
8, 16, 32, 64, 128, 256
```

### Best Results

All Part 3 runs use 8 threads.

| Matrix Size | Best Block Size | Bandwidth | Speedup vs. Naive | Speedup vs. Row-wise |
|---:|---:|---:|---:|---:|
| 4096 | 8 | 8.4750 GB/s | 5.449x | 4.08x |
| 8192 | 8 | 9.1989 GB/s | 8.917x | 5.53x |
| 16384 | 16 | 4.5821 GB/s | 8.944x | 9.09x |

"Naive" is the Part 2 Numba index-recalculation kernel (`reorder_index_numba`) applied to the same matrix. "Row-wise" is the untiled control. The speedup over row-wise isolates the locality benefit from the cheaper index arithmetic.

Blocking substantially improves locality for the rank-2 transpose. The best tile size depends on the matrix size and cache behavior.

The observed performance cliffs are consistent with cache-capacity and cache-associativity effects, especially because the tested leading dimensions are powers of two. Hardware performance counters were not collected, so this is an interpretation rather than a direct measurement.

For Rank-N tensors, the same idea can be generalized by dividing the N-dimensional iteration space into hyper-rectangular tiles and processing the corresponding source and destination tiles together. The source's last dimension and destination's first dimension are particularly important because they are the fastest-varying dimensions in their respective layouts.

Detailed Part 3 analysis is in:

```text
HW2_part3_analysis.md
runs/part3_blocking.csv
runs/part3_blocking.txt
```

---

## Output Files

Benchmark CSV files contain the three measured runtimes, the median runtime, the modeled traffic, the effective bandwidth, and a speedup column.

The main Part 2 CSV columns are below. Here `speedup` is relative to the 1-thread run of the same rank.

```text
rank
shape
threads
run1
run2
run3
median_time
traffic_bytes
bandwidth_gbs
speedup
```

The Part 3 CSV columns are below. `method` is one of `naive`, `rowwise`, or `blocked`; `block_size` is empty for the untiled methods; and `speedup_vs_naive` is relative to the naive run of the same matrix size.

```text
matrix_size
method
block_size
threads
run1
run2
run3
median_time
traffic_bytes
bandwidth_gbs
speedup_vs_naive
```

---

## Generative AI Usage

Generative AI was used during development for code review, debugging assistance, experiment planning, and analysis.

The required AI usage documentation, prompts, and transcripts are provided in the `ai/` directory.
