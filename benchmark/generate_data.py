"""Generate the shared benchmark data.

Writes raw Float64 binaries into `benchmark/data/` that both the Python and the
Julia benchmark load, so both implementations run on bit-identical input:

  <case>.data.bin      column-major (channels, times, subjects) data;
                       Python reshapes the buffer to (channels, times*subjects),
                       Julia reads it directly as (channels, times, subjects)
  <case>.adjacency.bin column-major (channels, channels) symmetric matrix

Nothing fancy: three problem sizes, sensor-space maps over channels with a
random 2-D channel layout, a mix of positive and negative blobs over noise.
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

CASES = [
    ("small", 32, 128, 10),
    ("medium", 64, 256, 20),
    ("large", 128, 512, 30),
]

# keep the average degree of the random layout roughly constant across sizes
RADIUS = {name: 0.35 * np.sqrt(32 / C) for (name, C, T, S) in CASES}


def random_layout_adjacency(rng, C, radius):
    pos = rng.random((C, 2))
    dist = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    adj = (dist < radius).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    return adj


def make_case(rng, name, C, T, S):
    d = rng.standard_normal((C, T, S))

    # add a positive or negative blob (a run of adjacent-ish channels) to ~40%
    # of the columns, so both sides of two_sided have real clusters to chew on
    for t in range(T):
        for s in range(S):
            r = rng.random()
            if r < 0.25:
                c0 = int(rng.integers(0, C - 12))
                d[c0 : c0 + int(rng.integers(5, 13)), t, s] += 2.5
            elif r < 0.40:
                c0 = int(rng.integers(0, C - 12))
                d[c0 : c0 + int(rng.integers(5, 13)), t, s] -= 2.5

    adj = random_layout_adjacency(rng, C, RADIUS[name])

    # column-major (first axis fastest) on disk, so Julia can read! it directly
    np.ascontiguousarray(d.T).tofile(DATA / f"{name}.data.bin")
    np.ascontiguousarray(adj.T).tofile(DATA / f"{name}.adjacency.bin")

    n_pos = int((d > 0).sum())
    print(f"{name}: channels={C} times={T} subjects={S}  positive elements: {n_pos}")


def main():
    DATA.mkdir(exist_ok=True)
    rng = np.random.default_rng(2026)
    for (name, C, T, S) in CASES:
        make_case(rng, name, C, T, S)


if __name__ == "__main__":
    main()
