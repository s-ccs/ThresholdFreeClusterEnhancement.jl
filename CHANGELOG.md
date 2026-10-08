# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [Unreleased]

- New `permutation_test` entry point: a one-sample sign-flip permutation test of
  a TFCE score map, built on [PermutationTests.jl]. The permutation mechanism is
  the package's `OneSampStatistic` (`StudentT_1S`) sign-flip scheme, driven by a
  custom `TFCEOneSampT` statistic; returns FWE-corrected and uncorrected
  p-value maps, each shaped `channels × times`. New runtime dependency:
  `PermutationTests`.
- New `tails.jl`: Winkler et al. (2016) tail approximations ported from the
  reference toolbox's `tfce/tails.py` — a Gamma moment-fit of the
  maximum-statistic null for FWE p-values (`gamma_pvalue`) and a Generalised
  Pareto fit (probability-weighted moments, pooled shape, Anderson–Darling
  acceptance gate) of each element's own permutation tail for uncorrected
  p-values (`pareto_pvalue`). Both fall back to counting when a fit degenerates,
  so p-values resolve below the `1/nperm` counting floor.
- Test items `test/test-permutation.jl` (tagged `:permutation`): permuted
  statistic vs. an independent naive refit, exact `2^n` enumeration vs.
  Monte-Carlo counting, tail-fit behaviour (mirrors the reference
  `test_tails.py`), null calibration (uniform uncorrected p, FWE control), and
  cross-language parity against the Python reference (fixtures in
  `test/parity/`, tagged `:crosslang`).
- `permutation_test`: the `:count` method now uses the
  `(count + 1) / (nperm + 1)` convention for both p-value maps — the observed
  value counts as one of the `nperm + 1` exchangeable values — so counting
  p-values never reach `0` and resolve down to `1/(nperm + 1)`.
- `permutation_test`: the Monte-Carlo null no longer copies the per-element
  data first (its working buffer is the one the observed statistic was taken
  from), and the per-element top-`K` null values are only gathered into the
  dense tail matrix when `method = :tail`.
- Initial release
- `src/` split into per-topic files (`adjacency.jl`, `workspace.jl`, `sort.jl`,
  `sweep.jl`, `tfce.jl`); `ThresholdFreeClusterEnhancement.jl` is the module shell
- New `sortalg` keyword: `:quick` (default, `Base` quicksort) or `:bucket`
  (experimental exponent-bucketed counting sort, ~10% faster on large maps)

<!-- Links -->

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[permutationtests.jl]: https://github.com/Marco-Congedo/PermutationTests.jl
[semantic versioning]: https://semver.org/spec/v2.0.0.html

<!-- Versions -->

[unreleased]: https://github.com/s-ccs/ThresholdFreeClusterEnhancement.jl/compare/v0.1.0...HEAD
