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

Julia 1.12.7, 8-core host. "Before" is the original barebone port, kept verbatim
as a second module and loaded in the *same process* as the optimized one, with
the two timed **interleaved** (best of 31 each, alternating). Interleaving matters
here: the host drifts by several percent over the course of a session, so two
separately-timed runs are not comparable. The repo's own `benchmark.jl`
(best of 3, un-interleaved) is quoted alongside as a sanity check.

| case   | shape (channels × times × subjects) | elements  | Julia before | Julia after | speedup | `benchmark.jl` | Python/C |
|--------|-------------------------------------|-----------|--------------|-------------|---------|----------------|----------|
| small  | 32 × 128 × 10                       | 40,960    | 11.00 ms     | 3.26 ms     | 3.38×   | 3.23 ms        | 5.48 ms  |
| medium | 64 × 256 × 20                       | 327,680   | 91.96 ms     | 24.91 ms    | 3.69×   | 25.49 ms       | 37.56 ms |
| large  | 128 × 512 × 30                      | 1,966,080 | 584.37 ms    | 166.47 ms   | 3.51×   | 171.37 ms      | 219.52 ms |

A second independent interleaved run gave 3.41× / 3.67× / 3.44×; an earlier one
(before the branchless gather, below) gave 2.96–3.34× / 3.07–3.23× /
2.99–3.13×. The "after" column is stable to ~1% but the "before" column is not,
because the original allocates hard enough that its best-of-N depends on where
the collector happens to be. **≈3.5× is the honest single figure** (it was ≈3.0×
before the last change).

Max |julia − python| over all elements: **7.63e-6** (float32 output rounding) —
the two implementations agree to output precision, before and after.

The optimized port is **1.32–1.68× ahead** of the reference C core (it started
2.5× behind), and it is **bit-for-bit identical** to the original Julia port on
84 randomized and hand-computed cases — hand-computed tiny maps, one-sided and
two-sided variants, `E = 0`, integer/`Float32`/`Symmetric`/sparse/`Int`
adjacency, sizes straddling every sort dispatch threshold, maps spanning
hundreds of binades (the wide-span fallback), geometric ladders, and single
channel / single column / all-negative / negative-zero edge cases. Seven of the
84 are the same map at `nthreads` = 2, 3, 4, 7, 8 and 16, each bit-identical to
`nthreads = 1`.

## What changed

All of it came out of a Profile.jl flat profile, and all of it is
output-preserving.

- **A reusable `Workspace` per thread.** The original allocated a dozen N-sized
  arrays per map (the `alive = [Int[] for _=1:N]` alone was 11% of runtime) and
  grew the max-tree node arrays with `push!` per node. Everything is now
  allocated once and reused. `Profile.Allocs` at `sample_rate = 1.0` counts
  **54 / 55 / 58 allocations** for the small / medium / large cases — a count
  that does not move when the number of maps goes from 1,280 to 15,360, i.e.
  **0 allocations per map**. The total is one output array plus one workspace.
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
- **A counting sort on the exponent field.** The positive part of a map is
  sorted as 16-byte `(UInt64, Int32)` key/index records. The key is the raw
  positive-`Float64` bit pattern, so the pattern *is* `exponent << 52 | mantissa`
  and all keys sharing an exponent are contiguous in sorted order — bucketing on
  the exponent is an **exact partition, not a quantisation**. Exponent spans
  inside a map are tiny (measured on this data: median 5–8, max 22), so the
  histogram is cheap to clear and buckets hold a handful of keys; only those few
  need an insertion sort. Above 40 keys per map this beats both a comparison
  sort and plain insertion sort, and unlike insertion sort it barely grows with
  `n` (0.26 ms per 625 maps at m=32, 0.27 ms at m=64, 0.38 ms at m=128, against
  0.64/0.64/1.30 for insertion sort) — 1.42× over `Base.Sort.QuickSort` at
  m=64, 1.48× at m=128. Below 40 keys insertion sort is kept: the counting
  sort's fixed costs do not pay for themselves. Measured against
  `Base.Sort.QuickSort`, insertion sort is itself 1.03× at m=16, 1.24× at m=32
  and 1.14× at m=64 — so `QuickSort` is never the best choice at any size, and
  is kept only as the fallback for data spanning more than 24 binades. It is the
  one Base algorithm that does not allocate; `sort!`'s default resolves to a
  `ScratchQuickSort` that mallocs a scratch buffer on *every* call, ~1.7× slower.
- **A hoisted parent pointer in the union-find.** The neighbour loop already
  loads `parent[v]` to test whether a merge would be a no-op; that load is handed
  to the `find` that the merge then performs, saving a redundant read. Measured
  interleaved: 1.7% (small), 0% (medium), 2.9% (large).
- **A branchless gather.** `x > 0.0` is a coin flip on noise-like data, so
  `x > 0.0 || continue` mispredicts about half the time and the profiler charged
  the gather 12% of the large case. Writing the `(key, index)` record
  unconditionally and folding the predicate into the counter with
  `ifelse` removes the branch; the slot written when the test fails is at most
  `ord[N]` and is never read, because only `ord[1:n]` is handed to the sort.
  Interleaved best-of-15, confirmed over two runs: **1.11–1.13× (small),
  1.11–1.13× (medium), 1.13–1.15× (large)** — the gather itself drops from 12%
  of the large case to 2%. Specialising the loop separately for `sgn = ±1` to
  drop the multiply was *also* tried and was consistently slightly slower than
  this (1.09–1.10× vs 1.11–1.15×), so it was not kept.

`level` is deliberately never reset between maps, so stale `root_stamp` entries
always compare as older and the O(N) reset is skipped; `active` is cleared only
for the elements each pass actually set.

## What did not work

Recorded because the numbers are the interesting part:

- **Radix / MSD sorting on quantised heights.** Rejected on correctness grounds
  (a quantised key is not a total order) and, as a byte-radix, on speed: positive
  doubles from Gaussian data cluster in a narrow exponent range, so a top-byte
  radix lands almost everything in 2–3 buckets. The exponent-bucket counting
  sort above is the version of this idea that *is* exact.
- **Encoding the activity flag as `parent[v] == 0`**, dropping the separate
  `active` array so the hottest loop touches one array instead of two.
  Interleaved best-of-15: 1.3% (small), 0.5% (medium), **−0.2% (large)** — a
  wash. The 128-byte `active` array lives in L1 regardless, so the saved load
  buys nothing, and the extra compare offsets it. Reverted; the clearer separate
  array stays.
- **Free path compression** (`parent[u] = ru` after the neighbour walk) and
  **fusing the activate and merge loops** (which would also have silently
  narrowed behaviour for non-symmetric adjacency). Both measured slower.
- **Replacing the per-column `view` with parent-array + offset indexing.** The
  profile shows `multidimensional.jl`/`range.jl`/`abstractarray.jl` frames
  around the column loop, which look like `SubArray` indexing overhead. They are
  not: measured directly on 15,360 columns of 128, `view(maps, :, b)[i]` in a
  `@inbounds` loop takes 1.602 ms against 1.603 ms for `maps[i + off]`, and
  `out[u] += x` takes 0.279 ms against 0.281 ms; building the two views per
  column costs 0.031 ms against 0.032 ms for the offset arithmetic. Identical
  within noise, so the views are already compiled away and there is nothing to
  win — those frames are profiler landing-pad artefacts.

## Where the time goes now

Flat profile of the large case (128 channels, 1 thread, 40 reps, single thread
filtered), as a share of the samples inside `tfce`. These are *inclusive* shares
measured at the phase anchors, so the nested rows overlap and do not sum to 100%:

| phase | share |
|-------|-------|
| neighbour scan + union-find (`_merge!`/`_findroot!`, lines 403–419, 465–505) | ~50% |
| sort (line 363 dispatch; 74% of it the intra-bucket insertion sorts) | 19% |
| &nbsp;&nbsp;of which insertion sorts inside the exponent buckets | 12% |
| &nbsp;&nbsp;of which scatter into buckets | 1.7% |
| &nbsp;&nbsp;of which histogram + exponent min/max scan | 1.5% |
| `pow` once per level (line 389) | 4.8% |
| find-root in the node-creation loop (line 419) | 4.7% |
| gather the positive part (branchless now) | 2.2% |
| accumulate into `out` (line 453) | 0.8% |

The gather used to be 12% and is now 2%; the union-find now dominates and is
inherent to the algorithm — the C core does the same work per
`(active element, neighbour)` pair, and the profile shows the cost is in the
loads and the compare at line 412, not in `_findroot!` itself. The sort is close
to its floor for this approach: three quarters of what is left is the insertion
sort inside a handful of two-element buckets, which is why the radix variants
above do not win.

## Threading

`tfce(...; nthreads = n)` is **opt-in**; the default of `1` is the single-
threaded path above. Map columns never interact, so the split is exact — the
result is bit-identical for any thread count (checked for n = 1, 2, 3, 4, 7, 8,
16). Measured interleaved best-of-15, on an otherwise idle host:

| case   | 1 thread | 2 threads | 4 threads | 8 threads | 8-thread speedup |
|--------|----------|-----------|-----------|-----------|------------------|
| small  | 3.24 ms  | 1.69 ms   | 0.91 ms   | 0.73 ms   | 4.45× |
| medium | 24.84 ms | 12.83 ms  | 6.95 ms   | 3.70 ms   | 6.71× |
| large  | 165.85 ms| 84.65 ms  | 44.27 ms  | 22.14 ms  | 7.49× |

The small case scales least well, and is the noisiest point on the host: at 32
channels a whole map is a handful of cache lines, so the run is short enough that
the `Core.Box` sharing and barrier costs of `Threads.@threads` start to matter,
and below ~1 ms per-call overhead is a visible fraction of the total. Note the
loop body has to be a single call: `@threads` inlines its body into a closure
that shares every variable it *assigns* with all other workers through a
`Core.Box`, and two workers writing one workspace corrupt the union-find forest.

The 2- and 4-thread points were stable to ~2% across runs; the 8-thread column
should be read as "≈4.5–7.5× when the host is quiet", and is depressed if
anything else holds a core.
