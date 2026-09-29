# Benchmark

Compares the barebone Julia TFCE against the reference Python/C implementation
(`~/tfce/tfce/python`) on bit-identical data, both as-is and single-threaded
(`n_jobs=1` is the Python default; the Julia code is single-threaded too).
Nothing fancy: three problem sizes of sensor-space maps over channels, where
every (time, subject) column is an independent 1-D map under an explicit
channel adjacency — the computation both implementations share.

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

## Results (single-threaded, best of 3, same machine)

| case   | shape (channels × times × subjects) | elements | Julia   | Python/C |
|--------|-------------------------------------|----------|---------|----------|
| small  | 32 × 128 × 10                       | 40,960   | 12.8 ms | 5.5 ms   |
| medium | 64 × 256 × 20                       | 327,680  | 100 ms  | 37 ms    |
| large  | 128 × 512 × 30                      | 1,966,080| 575 ms  | 220 ms   |

The barebone Julia port lands ~2.5× behind the reference C core, which
preallocates per-thread workspaces and sorts with counting/radix sort instead
of allocating per map. Max |julia − python| over all elements: 7.6e-6 (float32
output rounding) — the two implementations agree to output precision.
