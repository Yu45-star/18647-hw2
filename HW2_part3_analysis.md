# HW2 Part 3 Blocking / Locality Analysis

## Goal

Part 2 showed that the rank-2 case (a matrix transpose) reaches only 0.3639 GB/s at 8 threads: source reads are contiguous, but every destination write is `N` bytes away from the previous one. Part 3 explores whether blocking (tiling) the iteration space improves locality, starting from the rank-2 case.

## Implementation

All Part 3 code is separate from the Part 2 implementation, which is unchanged and serves as the naive baseline.

- `src/blocked_reorder.py`
  - `reorder_blocked_numba(src, dst, shape, block)`: Numba-parallel blocked transpose. The `rows × cols` matrix is divided into `B × B` tiles; `prange` iterates over tiles, and all elements inside one tile are written to their transposed destination, `dst[c * rows + r] = src[r * cols + c]`. Boundary tiles are clipped to the matrix edge.
  - `reorder_rowwise_numba(src, dst, shape)`: an untiled control with the same two-loop structure (one full source row at a time).
- `src/test_blocked_reorder.py`: correctness tests.
- `src/benchmark_blocking.py`: benchmark script.

The control is needed because the Part 2 kernel decodes every index with integer division/modulo, while the blocked kernel uses plain nested loops. Comparing blocked against naive therefore mixes two effects. Comparing blocked against rowwise isolates the locality effect, since both have the same cheap index arithmetic.

## Correctness

NumPy's transpose is used only as a test oracle. The tests cover block sizes `1, 8, 16, 32, 64, 128, 256` on the following shapes:

| Case | Shapes |
|---|---|
| Divisible by the block size | `16×16`, `64×64`, `256×256` |
| Not divisible by the block size (boundary tiles) | `17×17`, `100×100`, `300×300` |
| Smaller than the block | `1×1`, `5×5` |
| Rectangular | `33×65`, `128×40`, `1×37`, `37×1` |

Each case runs with 1, 2 and 4 threads, using both `int64` data (every element unique) and `uint8` data. The blocked output is also checked element by element against the Part 2 `reorder_index_numba` on `257×257`, `512×384` and `1000×1000` random `uint8` data. All tests pass, and the existing `src/test_reorder.py` still passes.

## Benchmark setup

```bash
cd src
python test_blocked_reorder.py
python benchmark_blocking.py --threads 8
```

The benchmark ran on the same AWS machine as Part 2 (Intel Xeon Platinum 8488C, 8 physical cores), with Numba 0.60.0 and NumPy 2.0.2, using 8 threads so that block size is the main independent variable. The methodology matches Part 2:

- `np.uint8` data;
- one warm-up call before timing, so JIT compilation and first-touch page faults are excluded;
- each configuration timed three times, reporting the median;
- the same traffic model as Part 2:

\[
\text{Traffic} = 2 \times N^2 \text{ bytes}, \qquad
\text{Bandwidth} = \frac{\text{Traffic}}{\text{Median Runtime}}.
\]

Matrix sizes are `4096²` (16 MiB), `8192²` (64 MiB) and `16384²` (256 MiB) per array, with block sizes `8, 16, 32, 64, 128, 256`. The raw results are in `runs/part3_blocking.csv` and `runs/part3_blocking.txt`.

## Results

Effective bandwidth (GB/s) at 8 threads:

| N | naive | rowwise | B=8 | B=16 | B=32 | B=64 | B=128 | B=256 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1.5555 | 2.0774 | **8.4750** | 3.9720 | 4.3046 | 4.2844 | 4.3585 | 4.1832 |
| 8192 | 1.0316 | 1.6645 | **9.1989** | 4.0453 | 4.5817 | 4.3870 | 4.2239 | 4.0484 |
| 16384 | 0.5123 | 0.5043 | 3.6421 | **4.5821** | 4.2538 | 4.1770 | 3.9590 | 1.8520 |

Best block size per matrix size:

| N | Best B | Bandwidth (GB/s) | Speedup vs. naive | Speedup vs. rowwise |
|---:|---:|---:|---:|---:|
| 4096 | 8 | 8.4750 | 5.449× | 4.08× |
| 8192 | 8 | 9.1989 | 8.917× | 5.53× |
| 16384 | 16 | 4.5821 | 8.944× | 9.09× |

The best measured bandwidth is 9.1989 GB/s (`8192²`, `B = 8`), about 25× the Part 2 rank-2 result at `65536²` (0.3639 GB/s).

## Interpretation

### Blocking improves bandwidth, and the gain is due to locality

Every blocked configuration is faster than the naive kernel, by 2.6× to 8.9×. The one exception is `B = 256` at `N = 16384`, which still gives 3.6×.

- At `N = 4096` and `8192`, rowwise is 1.3–1.6× faster than naive. That part of the gain comes from avoiding the per-element division/modulo.
- At `N = 16384`, rowwise and naive are identical (0.50 vs. 0.51 GB/s). The kernel is fully memory-bound there, and the 9.1× gain of `B = 16` over rowwise comes entirely from improved locality.
- Rowwise bandwidth collapses as `N` grows (2.08 → 1.66 → 0.50 GB/s), while the best blocked bandwidth degrades much less (8.48 → 9.20 → 4.58 GB/s).

### Very small and very large blocks perform worse

- **4096 and 8192:** `B = 8` is about 2× faster than every larger block. Blocks from 16 to 256 form a plateau of roughly 4.0–4.6 GB/s.
- **16384:** the optimum moves to `B = 16`, and `B = 8` drops below the plateau. `B = 256` collapses to 1.85 GB/s.

### A cache-associativity explanation

No hardware counters were collected, so the following is an interpretation that is consistent with the data, not a direct measurement. The key factor is that all tested `N` are powers of two, so the destination rows of a tile are separated by a power-of-two stride. The relevant per-core caches of the 8488C are:

| Cache | Size | Associativity | Way size (same-set address stride) |
|---|---:|---:|---:|
| L1D | 48 KiB | 12-way | 4 KiB |
| L2 | 2 MiB | 16-way | 128 KiB |

1. **L1 conflicts explain the cliff between B=8 and B=16.** With `N ≥ 4096`, the `B` destination cache lines of one tile are `N` bytes apart, a multiple of 4 KiB, so they all map to the same L1 set. Eight lines fit in the 12 ways. Sixteen or more lines evict each other, and the tile is effectively served from L2. This matches the flat ~4.3 GB/s plateau for all `B ≥ 16`.
2. **L2 conflicts explain the collapse of B=256 at N=16384.** Lines spaced `N` bytes apart can occupy at most about `2 MiB / N` lines of L2. That limit is 512 lines for `N = 4096`, 256 for `N = 8192` and only 128 for `N = 16384`. A `B = 256` tile needs 256 resident destination lines, which exceeds the limit only at `N = 16384`. That is the only case where `B = 256` breaks down.
3. **Partial-line write amplification explains the drop of B=8 at N=16384.**
   - A tile writes only `B/64` of each 64-byte destination line. The rest of the line is written `64/B` tile rows later, and in between each thread writes about `N` other lines.
   - For smaller `N`, those lines survive in L2/L3 until they are completed.
   - At `N = 16384`, the combined per-tile-row footprint of 8 threads is about 8 MiB. Lines are likely evicted before they are fully written, so each line is read for ownership and written back up to 8 times with `B = 8`, but only 4 times with `B = 16`. This favors `B = 16`.

A smoke run on a local Apple M3 (not used for the results) showed the same pattern: `B = 8`/`16` were best, and larger blocks were clearly slower.

### Evidence of reduced cache/TLB pressure

The evidence is indirect:

- Untiled bandwidth falls sharply with `N`, while blocked bandwidth stays several times higher.
- The optimal block size and the sizes at which cliffs appear match the capacity and associativity model above.

TLB effects cannot be separated from cache effects here: every destination row occupies at least one 4 KiB page, so a tile always touches `B` destination pages. Direct confirmation would require hardware counters, for example `perf stat -e L1-dcache-load-misses,dTLB-store-misses,LLC-load-misses`.

### Anomalies and measurement stability

- `4096²` naive: run 1 (0.0273 s) is slower than runs 2–3 (~0.0214 s). This is first-run noise, removed by the median.
- `8192²` naive and rowwise: about 15% spread between runs.
- `16384²` with `B = 256`: about 7% spread.
- All other configurations vary by less than 2%. The roughly 2× gap between `B = 8` and `B = 16` at `N ≤ 8192` is consistent across all three runs, so it is not noise.

Two follow-up experiments would strengthen the interpretation:

1. Rerun with non-power-of-two sizes (e.g. `--sizes 4160 8256`). If the associativity explanation is correct, the plateau for `B ≥ 16` should disappear.
2. Test nearby block sizes (e.g. `--blocks 4 12`). The optimum `B = 8` lies at the edge of the tested range, so this checks whether it moves lower.

## Generalization to Rank-N reorder

The same idea extends to the Rank-N reorder:

1. Divide the N-dimensional iteration space into hyper-rectangular tiles.
2. Process one source tile and its corresponding reordered destination tile at a time.
3. Choose tile dimensions so that the working set fits in cache.

The dimensions that matter most are the fastest-varying dimension of the source and the fastest-varying dimension of the destination. Under the reversed layout used in this homework, these are the last and the first dimensions:

- **Source side:** a tile extent of `B_last` along the last dimension gives contiguous reads.
- **Destination side:** a tile extent of `B_first` along the first dimension gives contiguous writes.
- **Middle dimensions:** a tile extent of 1 is sufficient; they become outer loops.

The rank-N reorder therefore reduces to outer loops over the middle dimensions around a 2D blocked transpose core. The rank-2 experiment shows that `B_first × B_last` must respect both cache capacity and associativity. For power-of-two shapes, padding the leading dimension or staggering tile origins would relieve the set conflicts observed here.

## Conclusion

For the rank-2 case:

- Blocking raises effective bandwidth from 0.5–1.6 GB/s to up to 9.2 GB/s, a speedup of up to about 9× over the Part 2 implementation.
- The rowwise control shows that, for large matrices, the entire gain comes from improved locality rather than cheaper index arithmetic.
- The best block size is small (8–16) and depends on `N`, because the power-of-two row stride causes set-associativity conflicts in L1 and L2.

Tile dimensions therefore have to be chosen with both cache capacity and associativity in mind. The same principle carries over to blocked Rank-N reordering.
