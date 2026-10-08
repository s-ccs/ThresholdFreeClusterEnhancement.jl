# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2020-2026, Christian Gaser. See LICENSE.
"""
Fixtures for the cross-language parity test item.

Runs the reference toolbox from source and dumps, as plain float64 little-endian
`.bin` files, both the inputs and the reference outputs of the two things the
Julia package must match to floating-point precision:

1. `onesamp.bin` -- the one-sample t under a fixed sign-flip, as computed by the
   reference's `PermutedGLM.fit_signs` (the statistic the sign-flip scheme must
   reproduce).

2. `tails.bin` -- a real sign-flip permutation null (per-element t under random
   signs), plus the observed statistic, the maximum-statistic null, and the
   largest-100 per-element tails; the reference's `gamma_pvalue` and
   `pareto_pvalue` applied to them (the p-value machinery the Julia port must
   reproduce).

Every input is drawn from a fixed seed, so the fixtures are deterministic and
committed; regenerate them with

    PYTHONPATH=/home/agent/tfce/tfce/python/src \
        python test/parity/make_parity.py
"""

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "tfce" / "python" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tfce.glm import PermutedGLM
from tfce.tails import gamma_pvalue, pareto_pvalue

HERE = Path(__file__).resolve().parent

N_PERM = 512
M = 120            # elements
K = 100            # tail size


def _write(name: str, *arrays) -> None:
    """Write float64 arrays back to back into one little-endian .bin file."""
    with open(HERE / name, "wb") as f:
        for a in arrays:
            f.write(np.ascontiguousarray(a, dtype=np.float64).tobytes(order="C"))


def make_onesamp() -> None:
    rng = np.random.default_rng(100)
    n = 12  # subjects
    Y = rng.standard_normal((M, n))
    Y[5] += 2.0  # one clearly affected element
    signs = rng.choice([-1.0, 1.0], size=n)

    t_ref = PermutedGLM(Y, np.ones((n, 1)), [1.0]).fit_signs(signs)

    _write("onesamp.bin", Y, signs, t_ref)


def make_tails() -> None:
    rng = np.random.default_rng(200)
    n = 12  # subjects

    # a real permutation null: per-element one-sample t under sign-flips
    Y = rng.standard_normal((M, n))
    Y[20:40] += 0.7  # some elements carry a mild effect
    null = np.empty((N_PERM, M))
    for p in range(N_PERM):
        s = rng.choice([-1.0, 1.0], size=n)
        Ys = Y * s[None, :]
        m = Ys.mean(axis=1)
        sd = Ys.std(axis=1, ddof=1)
        null[p] = np.abs(m) / (sd / np.sqrt(n))

    null_max = null.max(axis=1)

    # observed statistics: each element sits at a different quantile of its own
    # null, from unremarkable (low q) to extreme (high q); the quadratic puts
    # enough elements in the low-exceedance regime that the GPD fit has to work
    q = 1.0 - 0.9999 * np.linspace(0.0, 1.0, M) ** 2
    stat = np.diagonal(np.quantile(null, q, axis=0))  # stat[e] at quantile q[e]

    cnt = (null >= stat[None, :]).sum(axis=0)
    tail = -np.sort(-null, axis=0)[:K]

    p_gamma_ref = gamma_pvalue(stat, null_max)
    p_pareto_ref = pareto_pvalue(stat, tail, cnt, N_PERM)

    # tail is written transposed (M x K, row-major) so Julia reads it as (K, M)
    # in its column-major layout: tail.T.tobytes() gives flat index e*K + k
    _write(
        "tails.bin",
        stat,
        null_max,
        tail.T,
        cnt,
        np.float64(N_PERM),
        p_gamma_ref,
        p_pareto_ref,
    )


def main() -> None:
    make_onesamp()
    make_tails()
    print(f"wrote {HERE / 'onesamp.bin'} and {HERE / 'tails.bin'}")


if __name__ == "__main__":
    main()
