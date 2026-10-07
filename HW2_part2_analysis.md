# HW2 Part 2 Performance Analysis

## Final benchmark setup

The final benchmark uses `2^32` `uint8` elements (4 GiB). For each valid square tensor rank (`1, 2, 4, 8, 16, 32`), the optimized Numba index-recalculation implementation was run with 1 through 8 threads on an 8-physical-core machine. Each configuration was measured three times and the median runtime was used.

The payload memory-traffic model counts one read and one write per element:

\[
\text{Traffic} = 2 \times 2^{32} = 8{,}589{,}934{,}592 \text{ bytes}.
\]

The reported effective bandwidth is

\[
\text{Bandwidth} = \frac{\text{Traffic}}{\text{Median Runtime}}.
\]

## Parallel scaling

The Numba implementation scales strongly across the 8 physical cores. At 8 threads, the measured speedups are:

| Rank | 8-thread speedup |
|---:|---:|
| 1 | 7.811× |
| 2 | 7.236× |
| 4 | 7.415× |
| 8 | 7.486× |
| 16 | 7.811× |
| 32 | 7.988× |

The near-linear scaling of the highest-rank cases indicates that those kernels are largely compute-bound: each thread has substantial independent index arithmetic to perform, so adding cores reduces runtime almost proportionally.

## Effective bandwidth across ranks

At 8 threads:

| Rank | Effective bandwidth (GB/s) |
|---:|---:|
| 1 | 4.3301 |
| 2 | 0.3639 |
| 4 | 0.3873 |
| 8 | 0.4157 |
| 16 | 0.2005 |
| 32 | 0.0892 |

Rank 1 has the highest effective bandwidth because its index calculation is trivial. High-rank tensors require more coordinate decoding and stride accumulation per element, including repeated integer division/modulo operations. Therefore, high-rank cases spend much more time on computation per byte moved.

Bandwidth is not strictly monotonic with rank. Ranks 2, 4, and 8 are close in bandwidth because the result depends on both arithmetic cost and the memory-access pattern created by each tensor shape. Different destination-write strides change cache and TLB locality.

## Roofline-style interpretation

The final results suggest two broad regimes.

Low-rank cases perform less index arithmetic and are relatively more sensitive to memory behavior. Rank 1 reaches 4.33 GB/s at 8 threads and still obtains 7.81× speedup.

High-rank cases have much more arithmetic per byte moved. Rank 32 reaches only 0.0892 GB/s at 8 threads, yet achieves 7.99× speedup. This combination—low effective bandwidth with nearly ideal core scaling—is consistent with a compute-bound kernel rather than a shared-memory-bandwidth bottleneck.

## Connection to Part 3

The rank-2 case is analogous to matrix transpose. Source reads are regular, but destination writes are highly strided, which reduces locality and increases cache/TLB pressure. A blocked implementation would operate on smaller sub-tensors so that source and destination working sets remain cache-resident for longer. This provides the natural motivation for the Part 3 blocking/locality exploration.
