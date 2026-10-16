"""Cross-check the Julia TFCE against MNE-Python's independent implementation.

MNE's `permutation_cluster_1samp_test` computes the one-sample t, sign-flip
permutations, and stepped (Riemann-sum) TFCE in pure NumPy/SciPy -- an
implementation independent of the reference C core the Julia package was
ported from and validated against.

The Julia package treats every (time, subject) column as an independent 1-D
map under the channel adjacency, with a *global* max-statistic FWE null. The
same convention is reproduced here in ONE call by flattening (channels x
times) into a single vector and using a block-diagonal adjacency (one C x C
channel-adjacency block per time point): clusters (and TFCE) never span time,
while H0 stays the maximum over all elements.

  python3 mne_check.py             # step 0.05 -> <case>.mne.{tstat,obs,p}.bin
  python3 mne_check.py 0.01 fine   # finer step -> <case>.mne.{obs,p}.fine.bin

`n_permutations="all"` runs the exact test (all 2^(S-1) - 1 flip patterns up
to the two-sided symmetry, deterministic), matching the Julia side's
`exact = true` (all 2^S patterns) -- the same null distribution, so the
p-value comparison carries no Monte-Carlo noise.
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import ttest_1samp

from mne.stats import permutation_cluster_1samp_test

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# the exact test needs few subjects; the small shared case only
CASES = {"small": (32, 128, 10)}


def to_csr(adj):
    """Full symmetric 0/1 adjacency as a scipy CSR array, without self-loops."""
    m = adj != 0
    np.fill_diagonal(m, False)
    indptr = np.zeros(m.shape[0] + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(m.sum(axis=1))
    indices = np.nonzero(m)[1].astype(np.int32)
    return sparse.csr_array(
        (np.ones(len(indices), dtype=np.float64), indices, indptr), shape=m.shape
    )


def main():
    step = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    if tag and not tag.startswith("."):
        tag = "." + tag

    for name, (C, T, S) in CASES.items():
        data = np.fromfile(DATA / f"{name}.data.bin", dtype=np.float64)
        data = data.reshape((C, T, S), order="F")
        adj = np.fromfile(DATA / f"{name}.adjacency.bin", dtype=np.float64)
        adj = adj.reshape((C, C), order="F")

        # observed t-map (the TFCE height), for the Julia-side check
        t_map = ttest_1samp(data, 0.0, axis=2).statistic
        np.ascontiguousarray(t_map.T).tofile(DATA / f"{name}.mne.tstat.bin")

        # one-sample test on the flattened (channels x times) map under the
        # block-diagonal adjacency (see module docstring). data.T is
        # (subjects, times, channels); reshaping flattens (times, channels)
        # with the channel fastest, i.e. Julia's column-major element order.
        X = data.T.reshape(S, C * T)
        adj_full = sparse.block_diag([to_csr(adj)] * T, format="coo")

        t0 = time.perf_counter()
        t_obs, _, p_values, H0 = permutation_cluster_1samp_test(
            X,
            threshold=dict(start=0, step=step),
            n_permutations="all",
            tail=0,
            adjacency=adj_full,
            out_type="mask",
            verbose=False,
        )
        dt = time.perf_counter() - t0

        # t_obs/p_values are flat in the channel-fastest element order, so a
        # Fortran-order reshape recovers the (channels, times) maps
        obs_map = np.abs(t_obs).reshape((C, T), order="F")
        p_map = p_values.reshape((C, T), order="F")
        np.ascontiguousarray(obs_map.T).tofile(DATA / f"{name}.mne.obs{tag}.bin")
        np.ascontiguousarray(p_map.T).tofile(DATA / f"{name}.mne.p{tag}.bin")

        n_thr = int(np.ceil(np.max(np.abs(t_map)) / step))
        print(
            f"{name}: step={step}  ~{n_thr} thresholds  {len(H0)} H0 values "
            f"(exact test)  max|t|={np.abs(t_map).max():.3f}  "
            f"max TFCE={obs_map.max():.1f}  {dt:.1f} s"
        )


if __name__ == "__main__":
    main()
