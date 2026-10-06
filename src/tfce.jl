"""
    tfce(data, adjacency; E = 0.5, H = 2.0, two_sided = true, nthreads = 1, sortalg = :quick)
        -> Array{Float32}

Threshold-free cluster enhancement (TFCE) of `data` over the first dimension (channels).

`data` is `channels × times × subjects` (a `channels × times` matrix or a single
`channels` vector also work); every column is treated as an independent 1-D map over
channels. `adjacency` is a symmetric `channels × channels` neighbourhood matrix
(0/1, `Bool`, or weights; zero diagonal) indicating which channels are neighbours.

For each element `v`, TFCE integrates cluster-extent support over thresholds,
`TFCE(v) = ∫ e_v(h)^E · h^H dh`, where `e_v(h)` is the size of the connected component
containing `v` when thresholding the map at `h`. The integral is computed exactly with a
max-tree + union-find sweep over the positive part of each map; with `two_sided = true`
the negative part is processed as a second, negated map. Defaults `E = 0.5, H = 2.0`
suit volumetric data; channel/time-course maps typically use `E = 1, H = 2`.

`nthreads > 1` splits the (independent) map columns over that many Julia threads, each
with its own preallocated workspace. The default of `1` keeps the result single-threaded;
the result is bit-identical either way, because columns never interact.

`sortalg = :quick` (the default) sorts each map's positive values with `Base`'s quicksort.
`sortalg = :bucket` selects an experimental exponent-bucketed counting sort (see sort.jl)
that is faster on maps with many hundreds of channels. The results are identical either
way, because ties are consumed as one level regardless of the order they are sorted in.

Returns scores with the same shape as `data`.
"""
function tfce(
    data::AbstractArray,
    adjacency::AbstractMatrix;
    E = 0.5,
    H = 2.0,
    two_sided = true,
    nthreads = 1,
    sortalg = :quick,
)
    sortalg in (:quick, :bucket) ||
        throw(ArgumentError("sortalg must be :quick or :bucket, got $(repr(sortalg))"))
    N = size(data, 1)
    (N == 0 || length(data) == 0) && return zeros(Float32, size(data))

    maps = reshape(data, N, :)            # (channels, times·subjects)
    ncols = size(maps, 2)
    out = zeros(Float32, size(maps))
    adj = csr_adjacency(adjacency)
    E = Float64(E)
    H = Float64(H)
    nworkers = min(max(Int(nthreads), 1), ncols)

    if nworkers == 1
        w = Workspace(N, E, sortalg)
        for b = 1:ncols
            _tfce_map!(view(out, :, b), view(maps, :, b), adj, w, H, two_sided)
        end
    else
        # One workspace per thread, indexed by `threadid()`: `:static` pins each chunk to
        # one thread, so two workers can never pick up the same workspace. The loop body
        # is a single call on purpose -- `@threads` inlines it into a closure that shares
        # every variable it *assigns* with all the other workers through a `Core.Box`,
        # and two workers writing one workspace corrupt the union-find forest.
        ws = [Workspace(N, E, sortalg) for _ = 1:Threads.maxthreadid()]
        Threads.@threads :static for t = 1:nworkers
            _tfce_columns!(
                out,
                maps,
                adj,
                ws[Threads.threadid()],
                H,
                two_sided,
                _column_chunk(ncols, nworkers, t),
            )
        end
    end
    return reshape(out, size(data))
end

# Columns `lo:hi` of `maps`, into the same columns of `out`, with workspace `w`.
function _tfce_columns!(out, maps, adj, w, H, two_sided, (lo, hi))
    for b = lo:hi
        _tfce_map!(view(out, :, b), view(maps, :, b), adj, w, H, two_sided)
    end
    return out
end

# Inclusive column range for worker `i` of `parts`: the first `r` workers get one
# extra column, and every worker gets a non-empty range for any `i <= parts <= n`.
function _column_chunk(n::Int, parts::Int, i::Int)
    q, r = divrem(n, parts)
    first = (i - 1) * q + min(i - 1, r) + 1
    return first, first + q - 1 + (i <= r ? 1 : 0)
end
