# Generative AI Prompts

Generative AI was used for planning, code review, debugging, experiment design, and documentation.

## Part 1 / Part 2

Representative prompts:

- Explain the assignment requirements and the difference between recursive, iterative, and index-recalculation approaches for Rank-N tensor reordering.

- Help design a Python implementation for arbitrary-rank tensor reordering without using NumPy transpose/permutation operations.

- Review the serial implementations and explain how to verify correctness using NumPy transpose only as a test oracle.

- Suggest a shared-memory parallelization strategy in Python and explain the limitations of multiprocessing and the Python GIL.

- Help optimize the index-recalculation implementation using Numba and `prange`, while preserving the same algorithmic approach.

- Design a benchmark for `2^32` `uint8` elements that tests all valid equal-dimension tensor ranks and thread counts from 1 to 8.

- Help interpret the measured bandwidth and speedup results, including the difference between memory-bound and compute-bound behavior.

## Part 3

Representative prompts:

- Explore how blocking/tiling used in matrix transpose can improve locality for the Rank-N tensor reordering problem.

- Implement a Numba blocked rank-2 transpose and compare it against both the original Part 2 implementation and an untiled row-wise implementation.

- Benchmark block sizes 8, 16, 32, 64, 128, and 256 on several matrix sizes, using 8 threads and three runs per configuration.

- Analyze whether the performance improvement comes from reduced index arithmetic or improved cache locality.

- Explain how the blocking idea could generalize from rank-2 transpose to Rank-N tensor reordering using hyper-rectangular tiles.

## Documentation

Representative prompts:

- Summarize the final Part 2 performance results and explain the observed scaling behavior.

- Write a concise Part 3 analysis that clearly separates measured results from inferred cache/TLB explanations.

- Organize the repository README with build/run instructions, experiment setup, key results, and AI-usage documentation.