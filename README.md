# ThresholdFreeClusterEnhancement

> [!WARNING]
> **This package was developed using a large language model (LLM).**
> The code, documentation, benchmarks, and tests were produced by an AI assistant
> and have not been thoroughly reviewed by human experts. Use at your own risk and
> independently verify any results before relying on them for research or practice.

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://s-ccs.github.io/ThresholdFreeClusterEnhancement.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://s-ccs.github.io/ThresholdFreeClusterEnhancement.jl/dev)
[![Test workflow status](https://github.com/s-ccs/ThresholdFreeClusterEnhancement.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/s-ccs/ThresholdFreeClusterEnhancement.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/s-ccs/ThresholdFreeClusterEnhancement.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/s-ccs/ThresholdFreeClusterEnhancement.jl)
[![Docs workflow Status](https://github.com/s-ccs/ThresholdFreeClusterEnhancement.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/s-ccs/ThresholdFreeClusterEnhancement.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![All Contributors](https://img.shields.io/github/all-contributors/s-ccs/ThresholdFreeClusterEnhancement.jl?labelColor=5e1ec7&color=c0ffee&style=flat-square)](#contributors)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

## Performance

This package is a Julia port of the reference Python/C TFCE implementation
[`tfce`](https://github.com/ChristianGaser/tfce) (Christian Gaser's
threshold-free cluster enhancement toolbox, also on
[PyPI](https://pypi.org/project/tfce/)).
Single-threaded wall time on three sensor-space problem sizes
(channels × times × subjects), same machine, same bit-identical input data
([full benchmark](benchmark/README.md)):

| case   | shape (channels × times × subjects) | elements  | `:quick` (default) | `:bucket` (flag) | Python/C |
|--------|-------------------------------------|-----------|--------------------|------------------|----------|
| small  | 32 × 128 × 10                       | 40,960    | 3.27 ms (1.68×)    | 3.10 ms (1.77×)  | 5.48 ms  |
| medium | 64 × 256 × 20                       | 327,680   | 26.50 ms (1.42×)   | 23.99 ms (1.57×) | 37.56 ms |
| large  | 128 × 512 × 30                      | 1,966,080 | 178.66 ms (1.23×)  | 156.02 ms (1.41×)| 219.52 ms |

Optional threading (`nthreads = n`) is bit-identical for any thread count and
scales to 7.6× on 8 threads for the large case.

Against MNE-Python's independent TFCE implementation (stepped Riemann-sum TFCE,
MNE's step 0.05–0.01), the Julia transform is **~400–2000× faster per map**,
with the Julia result exact at any setting; the exact permutation test is
~480× faster at matched accuracy (numbers: [benchmark/README.md](benchmark/README.md)).

**Correctness:** the Julia implementation produces **exactly the same results
as the [`tfce`](https://github.com/ChristianGaser/tfce) Python toolbox** — max |julia − python| over all ~2 million benchmark
elements is 7.63e-6, which is just the Python toolbox's float32 output rounding,
i.e. the two agree to output precision — and the Julia code itself is
bit-for-bit reproducible across runs and thread counts (84 randomized and
hand-computed checks).

**Correctness against MNE-Python:** an independent implementation check (MNE's
`permutation_cluster_1samp_test`, pure NumPy/SciPy, stepped Riemann-sum TFCE,
different authors from both the reference C core and this port) agrees at
MNE's discretization level — max |ΔTFCE| = 0.51 / 0.108 (relative 0.0077 /
0.0016) at MNE steps 0.05 / 0.01, shrinking ∝ step, Spearman ρ = 0.9995 of the
score maps — and the exact-test p-value maps agree to MNE's 1/511 counting
floor (max |Δp_fwe| = 0.002 at the finer step, Spearman ρ = 1.0). Protocol and
numbers: [benchmark/README.md](benchmark/README.md).

## Permutation testing

`permutation_test(data, adjacency; …)` runs a one-sample sign-flip permutation
test of a TFCE score map: for each (channel, time) element the one-sample
*t*-statistic across subjects is the height and TFCE over the channel
neighbourhood is the spatial statistic, under the sign-flip scheme of
[PermutationTests.jl] (the package's one non-stdlib dependency, used for the
permutation mechanism only). It returns two p-value maps, each
`channels × times` and two-sided in magnitude:

- `p_fwe` — FWE-corrected, from the null of the *maximum* statistic over all
  elements;
- `p_uncorrected` — from each element's own permutation null.

By default (`method = :tail`) the p-values come from the Winkler et al. (2016)
tail fits ported from the reference toolbox — a Gamma moment-fit for the
maximum-statistic null and a Generalised Pareto fit (pooled shape) for each
element's tail — so they resolve below the `1/nperm` counting floor; each fit
falls back to counting when it degenerates. `method = :count` uses plain
counting with the `(count + 1) / (nperm + 1)` convention (p-values never
reach `0`), and `exact = true` enumerates all `2^n` sign patterns (guarded to
`n ≤ 20` subjects). Reproducible under a fixed `seed`.

## Contributing

If you want to make contributions of any kind, please first that a look into our [contributing guide directly on GitHub](docs/src/90-contributing.md) or the [contributing page on the website](https://s-ccs.github.io/ThresholdFreeClusterEnhancement.jl/dev/90-contributing/)

---

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
