# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [Unreleased]

- Initial release
- `src/` split into per-topic files (`adjacency.jl`, `workspace.jl`, `sort.jl`,
  `sweep.jl`, `tfce.jl`); `ThresholdFreeClusterEnhancement.jl` is the module shell
- New `sortalg` keyword: `:quick` (default, `Base` quicksort) or `:bucket`
  (experimental exponent-bucketed counting sort, ~10% faster on large maps)

<!-- Links -->

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

<!-- Versions -->

[unreleased]: https://github.com/s-ccs/ThresholdFreeClusterEnhancement.jl/compare/v0.1.0...HEAD
