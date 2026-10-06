# Benchmark

Compares the Julia TFCE against the reference Python/C implementation
(`~/tfce/tfce/python`) on bit-identical data, both as-is and single-threaded
(`n_jobs=1` is the Python default; `tfce`'s `nthreads` default of `1` is
single-threaded too — see [Threading](#threading)). Nothing fancy: three problem
sizes of sensor-space maps over channels, where every (time, subject) column is
an independent 1-D map under an explicit channel adjacency — the computation
both implementations share.

## Files

- `generate_data.py` — writes the shared raw Float64 data + adjacency binaries
  into `data/` (git-ignored); both sides load these, so the input is identical.
- `benchmark.py` — times the reference implementation, writes each result to
  `data/<case>.out.python.bin` for the cross-check.
- `benchmark.jl` — times the Julia `tfce` on the same data and reports the
  maximum absolute difference against the Python output.

## Run

```sh
# reference environment: needs numpy + the reference package built
# (here: a venv with numpy/cython, extension compiled with zig cc)
python3 generate_data.py
PYTHONPATH=<reference>/python/src python3 benchmark.py
julia --project=. benchmark.jl
```

`E`, `H`, and `two_sided` are the shared defaults (0.5, 2.0, true) on both
sides, so the reported max-difference doubles as a correctness check.

## Results (single-threaded, same machine)

Julia 1.12.7, 8-core host. "original" is the initial barebone port, kept verbatim
as a second module and loaded in the *same process* as the current one, with all
variants timed **interleaved** (best of 21 each, alternating). Interleaving
matters here: the host drifts by several percent over the course of a session, so
two separately-timed runs are not comparable. The repo's own `benchmark.jl`
(best of 3, un-interleaved, `sortalg = :quick`) is quoted alongside as a sanity
check.

| case   | shape (channels × times × subjects) | elements  | original | `:quick` (default) | `:bucket` (flag) | `benchmark.jl` | Python/C |
|--------|-------------------------------------|-----------|----------|--------------------|------------------|----------------|----------|
| small  | 32 × 128 × 10                       | 40,960    | 10.06 ms | 3.27 ms (3.08×)    | 3.10 ms (3.25×)  | 3.25 ms        | 5.48 ms  |
| medium | 64 × 256 × 20                       | 327,680   | 81.20 ms | 26.50 ms (3.06×)   | 23.99 ms (3.38×) | 27.07 ms       | 37.56 ms |
| large  | 128 × 512 × 30                      | 1,966,080 | 602.75 ms| 178.66 ms (3.37×)  | 156.02 ms (3.86×)| 183.04 ms      | 219.52 ms |

The "original" column is noisy — the barebone port allocates hard enough that
its best-of-N depends on where the collector happens to be (an earlier
best-of-31 run gave 11.00 / 91.96 / 584.37 ms for it). The current columns are
stable to ~1%. **≈3.1–3.4× is the honest figure for the default**, ≈3.3–3.9×
with the `:bucket` flag.

Max |julia − python| over all elements: **7.63e-6** (float32 output rounding) —
the two implementations agree to output precision, for both `sortalg` values.

Both variants are **ahead of the reference C core**: 1.68× / 1.42× / 1.23× for
`:quick`, 1.77× / 1.57× / 1.41× for `:bucket` (the port started 2.5× behind).

Both are **bit-for-bit identical** to the original Julia port: on 84 randomized
and hand-computed cases (hand-computed tiny maps, one-sided and two-sided
variants, `E = 0`, integer/`Float32`/`Symmetric`/sparse/`Int` adjacency, sizes
straddling every sort threshold, maps spanning hundreds of binades, geometric
ladders, single channel / single column / all-negative / negative-zero edge
cases), and on every map of the benchmark data for both `sortalg` values. Seven
of the 84 are the same map at `nthreads` = 2, 3, 4, 7, 8 and 16, each
bit-identical to `nthreads = 1`.

## What changed in the simplification

The first optimization pass (below) left a single ~500-line source file with
`Int32` bookkeeping, a custom three-way sort dispatch, and micro-level hints.
This pass split it into per-topic files and dropped everything the profile said
was not paying for itself, with a new `sortalg` keyword so the fastest sort is
still available. `src/ThresholdFreeClusterEnhancement.jl` is now just
`module`/`using`/`export`/`include`; the code lives in `adjacency.jl`,
`workspace.jl`, `sort.jl`, `sweep.jl`, and `tfce.jl`.

- **The sort defaults to `Base.Sort.QuickSort` on plain `(Float64, Int)`
  records** — one line, no bit-pattern key, no `reinterpret` anywhere. Measured
  interleaved against the fully-optimized predecessor: 1.04× (small), 1.08×
  (medium), 1.11× (large) slower — the counting sort was worth ~10%, not the
  ~30% estimated from micro-benchmarks.
- **The counting sort is kept as `sortalg = :bucket`** (`sort.jl`). It is
  0.85–0.92× of the default (8–15% faster end-to-end), and with it the
  simplified code is marginally faster than the previous fully-optimized
  single-file version was (153.9 vs 164.4 ms on the large case, interleaved).
  Results are identical for both values: ties are consumed as one level
  regardless of the order they are sorted in.
- **Plain `Int` everywhere.** All `Int32`/`Int32(v)` casts, the `Int32`
  sentinel, and the `typemax(Int32)` overflow guard are gone (a 64-bit `level`
  cannot overflow in any conceivable run). Measured against an `Int32` twin of
  the same code: 1.02–1.03× — a few percent for a large readability win.
- **`parent` doubles as the activity flag** (`parent[u] == 0` ⟺ not yet
  activated this pass; it is `u` ⟺ u is its own root, so it is nonzero exactly
  while active). One array less in the workspace and one less load in the hot
  test; the earlier measurement called this a wash (below), and it stays one.
- **One `_findroot!` again.** The 3-argument form that took the caller's
  `parent[x]` load as a hint is back to a single 2-argument function with path
  halving; the hoisted-parent hint in `_merge!` is gone with it.
- **Kept:** the branchless gather (11–15% at every size, 3 lines), the `sizeE`
  table, the `level`-never-reset trick, the reusable per-thread `Workspace`, and
  the flat CSR adjacency.

## What changed in the first optimization pass

All of it came out of a Profile.jl flat profile, and all of it is
output-preserving.

- **A reusable `Workspace` per thread.** The original allocated a dozen N-sized
  arrays per map (the `alive = [Int[] for _=1:N]` alone was 11% of runtime) and
  grew the max-tree node arrays with `push!` per node. Everything is now
  allocated once and reused. `Profile.Allocs` at `sample_rate = 1.0` counts
  **51 / 51 / 53 allocations** for the small / medium / large cases — a count
  that does not move when the number of maps goes from 1,280 to 15,360, i.e.
  **0 allocations per map**.
- **A `k^E` lookup table** (`sizeE`) replaces `pow(size, E)` per max-tree node in
  the accumulation, and `h^(H+1)` is computed once per level instead of twice
  per node. This is where most of the win over C comes from — the C core still
  does two `pow` calls per node there.
- **A flat CSR adjacency** replaces `Vector{Vector{Int}}`, and the per-root
  `Vector{Int}` alive lists become an intrusive linked list (`head`/`tail`/
  `nd_next`) spliced in O(1) on union.
- **No per-column copies.** The original did `Float64.(view(maps, :, b))` and
  `-d` per map; the sign is folded into the comparison and the accumulation,
  which is exact because it is a ±1 multiply.
- **The branchless gather.** `x > 0.0` is a coin flip on noise-like data, so
  `x > 0.0 || continue` mispredicts about half the time and the profiler charged
  the gather 12% of the large case. Writing the record unconditionally and
  folding the predicate into the counter with `ifelse` removes the branch; the
  slot written when the test fails is at most `ord[N]` and is never read, because
  only `ord[1:n]` is handed to the sort. Interleaved best-of-15, confirmed over
  two runs: **1.11–1.13× (small), 1.11–1.13× (medium), 1.13–1.15× (large)**.
  Specialising the loop separately for `sgn = ±1` to drop the multiply was also
  tried and was consistently slightly slower (1.09–1.10×), so it was not kept.

`level` is deliberately never reset between maps, so stale `root_stamp` entries
always compare as older and the O(N) reset is skipped; `parent` is cleared only
for the elements each pass actually set.

### The `sortalg = :bucket` counting sort

The positive part of a map is sorted as `(Float64, Int)` records whose key is
compared *via the raw positive-`Float64` bit pattern*, so the pattern *is*
`exponent << 52 | mantissa` and all keys sharing an exponent are contiguous in
sorted order — bucketing on the exponent is an **exact partition, not a
quantisation**. Exponent spans inside a map are tiny (measured on this data:
median 5–8, max 22), so the histogram is cheap to clear and buckets hold a
handful of keys; only those few need an insertion sort. Above 40 keys per map
this beats both a comparison sort and plain insertion sort, and unlike insertion
sort it barely grows with `n` (0.26 ms per 625 maps at m=32, 0.27 ms at m=64,
0.38 ms at m=128, against 0.64/0.64/1.30 for insertion sort). Below 40 keys
insertion sort is kept: the counting sort's fixed costs do not pay for
themselves. A wide exponent span (data spanning many binades) falls back to
`QuickSort`, the one Base algorithm that does not allocate.

## What did not work

Recorded because the numbers are the interesting part:

- **Radix / MSD sorting on quantised heights.** Rejected on correctness grounds
  (a quantised key is not a total order) and, as a byte-radix, on speed: positive
  doubles from Gaussian data cluster in a narrow exponent range, so a top-byte
  radix lands almost everything in 2–3 buckets. The exponent-bucket counting
  sort above is the version of this idea that *is* exact.
- **Encoding the activity flag as `parent[v] == 0`.** Measured a wash back when
  the separate `active` array existed (interleaved best-of-15: 1.3% small, 0.5%
  medium, −0.2% large), so it was reverted then. The simplification pass adopted
  it anyway — for the one-array-less workspace, not for speed — and it is still
  a wash, in both `sortalg` variants.
- **Free path compression** (`parent[u] = ru` after the neighbour walk) and
  **fusing the activate and merge loops** (which would also have silently
  narrowed behaviour for non-symmetric adjacency). Both measured slower.
- **Replacing the per-column `view` with parent-array + offset indexing.** The
  profile shows `multidimensional.jl`/`range.jl`/`abstractarray.jl` frames
  around the column loop, which look like `SubArray` indexing overhead. They are
  not: measured directly on 15,360 columns of 128, `view(maps, :, b)[i]` in a
  `@inbounds` loop takes 1.602 ms against 1.603 ms for `maps[i + off]`, and
  `out[u] += x` takes 0.279 ms against 0.281 ms. Identical within noise — the
  views are already compiled away and those frames are profiler landing-pad
  artefacts.
- **Memoizing the once-per-level `pow`** (e.g. `Memoize.jl`): a dictionary lookup
  with a hashed `Float64` key costs more than the ~20–50 ns `pow` it would
  replace, t-stat heights essentially never repeat, and it would add a
  dependency. The `sizeE` table already removed the other `pow`.

## Where the time goes now

Flat profile of the large case (128 channels, `sortalg = :quick`, 1 thread, 40
reps, single thread filtered), as a share of the samples inside `tfce`. These
are *inclusive* shares measured at the phase anchors, so the nested rows overlap
and do not sum to 100%:

| phase | share |
|-------|-------|
| neighbour scan + union-find (hot test `sweep.jl:75` 699 samples self, `_merge!` call `:76` 490, `_merge!`/`_findroot!` 177/299) | ~50% |
| sort (`sort.jl:36` dispatch) | 28% |
| &nbsp;&nbsp;all of it `Base`'s quicksort on the `(Float64, Int)` records | — |
| `pow` once per level (`sweep.jl:52`) | ~4% |
| find-root in the node-creation loop (`sweep.jl:82`) | ~4% |
| gather the positive part (branchless) | <2% |
| accumulate into `out` | ~1% |

With `sortalg = :bucket` the sort drops back to ~19% (of which three quarters is
the insertion sort inside a handful of two-element buckets), which is the
difference between the two columns of the results table. The union-find dominates
either way and is inherent to the algorithm — the C core does the same work per
`(active element, neighbour)` pair.

## Threading

`tfce(...; nthreads = n)` is **opt-in**; the default of `1` is the single-
threaded path above. Map columns never interact, so the split is exact — the
result is bit-identical for any thread count (checked for n = 1, 2, 3, 4, 7, 8,
16). Measured interleaved best-of-15, on an otherwise idle host:

| case   | 1 thread | 2 threads | 4 threads | 8 threads | 8-thread speedup |
|--------|----------|-----------|-----------|-----------|------------------|
| small  | 3.28 ms  | 1.78 ms   | 0.97 ms   | 1.02 ms   | 3.2× (noisy) |
| medium | 26.47 ms | 13.51 ms  | 7.12 ms   | 4.35 ms   | 6.1× |
| large  | 178.60 ms| 89.91 ms  | 45.98 ms  | 23.51 ms  | 7.6× |

The small case scales least well, and is the noisiest point on the host: at 32
channels a whole map is a handful of cache lines, so the run is short enough that
the `Core.Box` sharing and barrier costs of `Threads.@threads` start to matter,
and below ~1 ms per-call overhead is a visible fraction of the total. The
8-thread column is the least reproducible (an adjacent run of the same script
gave 35.14 ms / 5.1× on the large case); read it as "≈5–7.6× when the host is
quiet". Note the loop body has to be a single call: `@threads` inlines its body
into a closure that shares every variable it *assigns* with all other workers
through a `Core.Box`, and two workers writing one workspace corrupt the
union-find forest.
