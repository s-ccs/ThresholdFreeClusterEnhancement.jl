"""Benchmark the reference sign-flip permutation test on the shared data.

Runs the reference toolbox's documented one-sample sign-flip permutation
workflow as-is (its ``docs/permutation.md``), single-threaded (``n_jobs=1``
is ``tfce``'s default): ``n_perm`` sign-flip permutations, each a
``fit_signs`` re-fit of the one-sample *t* map, TFCE over the channel
adjacency, and the documented accumulation of the maximum statistic (the
FWE null), the exact exceedance counts, and the top ``n_tail`` values of
each element's own null (overwrite the weakest, re-sort); then the
reference's ``gamma_pvalue`` and ``pareto_pvalue`` turn those into the two
p-value maps.

``E``, ``H``, and ``two_sided`` are the shared defaults (0.5, 2.0, true) on
both sides. Writes the observed score map and both p-value maps to
``data/<case>.perm.{obs,pfwe,punc}.python.bin`` so ``perm_benchmark.jl`` can
cross-check the two implementations.

The element order everywhere is the Julia side's: column-major over
``channels x times`` (channel fastest), hence ``.ravel(order="F")``.
"""

import time
from pathlib import Path

import numpy as np

from tfce import tfce
from tfce.glm import PermutedGLM
from tfce.tails import gamma_pvalue, pareto_pvalue

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

CASES = [
    ("small", 32, 128, 10),
    ("medium", 64, 256, 20),
    ("large", 128, 512, 30),
]

N_PERM = 1000
N_TAIL = 100
REPEATS = 3
SEED = 1234


def to_csr(adj):
    """(indptr, indices) of a symmetric 0/1 matrix, without self-loops."""
    m = adj != 0
    np.fill_diagonal(m, False)
    indptr = np.zeros(m.shape[0] + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(m.sum(axis=1))
    indices = np.nonzero(m)[1].astype(np.int32)
    return indptr, indices


def run_perm_test(data, csr, C, T, S, n_perm, seed):
    """The documented one-sample sign-flip permutation workflow, as-is."""
    M = C * T
    Y = data.reshape(M, S, order="F")  # elements x subjects, channel-fastest
    model = PermutedGLM(Y, np.ones((S, 1)), [1.0])

    # --- observed ---------------------------------------------------------
    # the flat element order is channel-fastest, so the (C, T) t-map needs a
    # Fortran-order reshape: [c, t] is the (channel c, time t) element
    t0 = model.fit(None).reshape(C, T, order="F")
    tfce0 = tfce(t0, adjacency=csr, E=0.5, H=2.0, two_sided=True)
    sgn = np.sign(tfce0.ravel(order="F"))  # both sides become an upper tail
    sgn[sgn == 0] = 1
    obs = np.abs(tfce0).ravel(order="F")

    # --- permutations -----------------------------------------------------
    null_max = np.empty(n_perm)  # for FWE
    tail = np.full((N_TAIL, M), -np.inf)  # for the uncorrected p-values
    cnt = np.zeros(M)  # exceedances, over ALL permutations

    rng = np.random.default_rng(seed)
    for p in range(n_perm):
        flips = rng.choice([-1.0, 1.0], size=S)
        tp = model.fit_signs(flips).reshape(C, T, order="F")
        tp = tfce(tp, adjacency=csr, E=0.5, H=2.0, two_sided=True)

        null_max[p] = np.abs(tp).max()

        v = tp.ravel(order="F") * sgn  # an upper tail on both sides
        cnt += v >= obs  # exact, and cheap

        # keep only the largest N_TAIL values per element, the simple way:
        # overwrite the weakest, re-sort
        tail[0] = v
        tail.sort(axis=0)

    p_fwe = gamma_pvalue(obs, null_max)
    p_unc = pareto_pvalue(obs, tail, cnt, n_perm)

    return tfce0, p_fwe, p_unc


def breakdown(data, csr, C, T, S):
    """Where one permutation's time goes: the library work vs the documented
    tail upkeep. 32 permutations, warm tail."""
    M = C * T
    Y = data.reshape(M, S, order="F")
    model = PermutedGLM(Y, np.ones((S, 1)), [1.0])
    tfce0 = tfce(model.fit(None).reshape(C, T, order="F"), adjacency=csr, E=0.5, H=2.0, two_sided=True)
    sgn = np.sign(tfce0.ravel(order="F"))
    sgn[sgn == 0] = 1
    obs = np.abs(tfce0).ravel(order="F")
    tail = np.full((N_TAIL, M), -np.inf)
    rng = np.random.default_rng(0)

    t_lib = t_tail = 0.0
    for _ in range(32):
        t0 = time.perf_counter()
        flips = rng.choice([-1.0, 1.0], size=S)
        tp = model.fit_signs(flips).reshape(C, T, order="F")
        tp = tfce(tp, adjacency=csr, E=0.5, H=2.0, two_sided=True)
        v = tp.ravel(order="F") * sgn
        _ = np.abs(tp).max()
        cnt_inc = (v >= obs).sum()
        t1 = time.perf_counter()
        tail[0] = v
        tail.sort(axis=0)
        t2 = time.perf_counter()
        t_lib += t1 - t0
        t_tail += t2 - t1

    return t_lib / 32 * 1000, t_tail / 32 * 1000


def main():
    for (name, C, T, S) in CASES:
        data = np.fromfile(DATA / f"{name}.data.bin", dtype=np.float64)
        data = data.reshape((C, T * S), order="F")  # (n_vertices, B) surface maps
        adj = np.fromfile(DATA / f"{name}.adjacency.bin", dtype=np.float64)
        adj = adj.reshape((C, C), order="F")
        csr = to_csr(adj)

        run_perm_test(data, csr, C, T, S, 20, 0)  # warm-up
        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            tfce0, p_fwe, p_unc = run_perm_test(data, csr, C, T, S, N_PERM, SEED)
            times.append(time.perf_counter() - t0)

        # column-major on disk, for the Julia cross-check
        np.ascontiguousarray(tfce0.T.astype(np.float32)).tofile(
            DATA / f"{name}.perm.obs.python.bin"
        )
        np.ascontiguousarray(p_fwe.reshape(C, T, order="F").T).tofile(
            DATA / f"{name}.perm.pfwe.python.bin"
        )
        np.ascontiguousarray(p_unc.reshape(C, T, order="F").T).tofile(
            DATA / f"{name}.perm.punc.python.bin"
        )

        best = min(times)
        print(
            f"{name}: C={C} T={T} S={S}  n_perm={N_PERM}  best: {best:8.2f} s  "
            f"runs (s): " + ", ".join(f"{t:.2f}" for t in times)
        )
        lib, tail = breakdown(data, csr, C, T, S)
        print(f"  per perm: library (signs+fit+tfce+max+cnt) {lib:6.2f} ms, "
              f"tail upkeep {tail:6.2f} ms")


if __name__ == "__main__":
    main()
