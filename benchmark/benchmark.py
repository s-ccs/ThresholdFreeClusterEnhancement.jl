"""Benchmark the reference Python/C TFCE on the same data the Julia side loads.

Runs the reference implementation as-is, single-threaded (``n_jobs=1``, its
default), on sensor-space maps over channels -- the same computation the Julia
package does: every (time, subject) column is an independent 1-D map over
channels under an explicit channel adjacency.

Data comes from `generate_data.py` (bit-identical to what the Julia side reads).
Writes each result to `data/<case>.out.python.bin` so the Julia script can
cross-check the two implementations agree.
"""

import time
from pathlib import Path

import numpy as np

from tfce.core import tfce

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

CASES = [
    ("small", 32, 128, 10),
    ("medium", 64, 256, 20),
    ("large", 128, 512, 30),
]

REPEATS = 3


def to_csr(adj):
    """(indptr, indices) of a symmetric 0/1 matrix, without self-loops."""
    m = adj != 0
    np.fill_diagonal(m, False)
    indptr = np.zeros(m.shape[0] + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(m.sum(axis=1))
    indices = np.nonzero(m)[1].astype(np.int32)
    return indptr, indices


def main():
    for (name, C, T, S) in CASES:
        data = np.fromfile(DATA / f"{name}.data.bin", dtype=np.float64)
        data = data.reshape((C, T * S), order="F")  # (n_vertices, B) surface maps
        adj = np.fromfile(DATA / f"{name}.adjacency.bin", dtype=np.float64)
        adj = adj.reshape((C, C), order="F")
        indptr, indices = to_csr(adj)

        tfce(data, adjacency=(indptr, indices))  # warm-up
        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            out = tfce(data, adjacency=(indptr, indices))
            times.append(time.perf_counter() - t0)

        # column-major (channels, times*subjects) on disk, for the Julia cross-check
        np.ascontiguousarray(out.T.astype(np.float32)).tofile(
            DATA / f"{name}.out.python.bin"
        )

        best = min(times) * 1000
        print(f"{name}: channels={C} cols={T*S}  best: {best:8.2f} ms  "
              f"runs (ms): " + ", ".join(f"{t*1000:.2f}" for t in times))


if __name__ == "__main__":
    main()
