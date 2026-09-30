module ThresholdFreeClusterEnhancement

using LinearAlgebra
using StatsBase

export tfce

# Sentinel for "no such node" in the per-root alive list (`head`/`tail`/`nd_next`).
const NONODE = Int32(-1)

# One map element queued for the sweep: the value's bit pattern and the channel it came
# from. A 64-bit key and a 32-bit index do not fit in one word, so they share a record
# (this is also the layout of the `VI` struct the C core sorts).
const KeyIndex = Tuple{UInt64,Int32}

"""
    tfce(data, adjacency; E = 0.5, H = 2.0, two_sided = true, nthreads = 1)
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

Returns scores with the same shape as `data`.
"""
function tfce(
    data::AbstractArray,
    adjacency::AbstractMatrix;
    E = 0.5,
    H = 2.0,
    two_sided = true,
    nthreads = 1,
)
    N = size(data, 1)
    (N == 0 || length(data) == 0) && return zeros(Float32, size(data))
    N <= typemax(Int32) ||
        throw(ArgumentError("only maps of up to $(typemax(Int32)) channels are supported"))

    maps = reshape(data, N, :)            # (channels, times·subjects)
    ncols = size(maps, 2)
    out = zeros(Float32, size(maps))
    adj = csr_adjacency(adjacency)
    E = Float64(E)
    H = Float64(H)
    nworkers = min(max(Int(nthreads), 1), ncols)

    if nworkers == 1
        w = Workspace(N, E)
        for b = 1:ncols
            _tfce_map!(view(out, :, b), view(maps, :, b), adj, w, E, H, two_sided)
        end
    else
        # One workspace per thread, indexed by `threadid()`: `:static` pins each chunk to
        # one thread, so two workers can never pick up the same workspace. The loop body
        # is a single call on purpose -- `@threads` inlines it into a closure that shares
        # every variable it *assigns* with all the other workers through a `Core.Box`,
        # and two workers writing one workspace corrupt the union-find forest.
        ws = [Workspace(N, E) for _ = 1:Threads.maxthreadid()]
        Threads.@threads :static for t = 1:nworkers
            _tfce_columns!(
                out,
                maps,
                adj,
                ws[Threads.threadid()],
                E,
                H,
                two_sided,
                _column_chunk(ncols, nworkers, t),
            )
        end
    end
    return reshape(out, size(data))
end

# Columns `lo:hi` of `maps`, into the same columns of `out`, with workspace `w`.
function _tfce_columns!(out, maps, adj, w, E, H, two_sided, (lo, hi))
    for b = lo:hi
        _tfce_map!(view(out, :, b), view(maps, :, b), adj, w, E, H, two_sided)
    end
    return out
end

"""
    adjacency_lists(adjacency) -> Vector{Vector{Int}}

Neighbour lists (without self-loops) of an adjacency matrix. Any `AbstractMatrix` works,
including `Symmetric`, `Diagonal`, and sparse wrappers.
"""
function adjacency_lists(adjacency::AbstractMatrix)
    adj = [Int[] for _ = 1:size(adjacency, 1)]
    for idx in findall(!iszero, adjacency)
        i, j = idx[1], idx[2]
        i == j && continue
        push!(adj[i], j)
    end
    return adj
end

# --------------------------------------------------------------------------------------
# Neighbourhood: explicit CSR, so the sweep walks a flat index array instead of chasing
# one small vector per channel.
# --------------------------------------------------------------------------------------
struct CSRAdjacency
    ptr::Vector{Int32}     # row u occupies ptr[u]+1 : ptr[u+1]
    idx::Vector{Int32}
end

function csr_adjacency(adjacency::AbstractMatrix)
    N = size(adjacency, 1)
    positions = findall(!iszero, adjacency)

    deg = zeros(Int32, N)                     # count first, then fill (two O(nnz) sweeps)
    for pos in positions
        pos[1] == pos[2] && continue
        deg[pos[1]] += Int32(1)
    end
    ptr = Vector{Int32}(undef, N + 1)
    ptr[1] = 0
    for u = 1:N
        ptr[u+1] = ptr[u] + deg[u]
    end
    idx = Vector{Int32}(undef, Int(ptr[N+1]))
    fill = copy(ptr)
    for pos in positions
        u, v = pos[1], pos[2]
        u == v && continue
        idx[fill[u]+1] = Int32(v)
        fill[u] += Int32(1)
    end
    return CSRAdjacency(ptr, idx)
end

# --------------------------------------------------------------------------------------
# Workspace: every per-pass array, allocated once per thread and reused across all maps.
# The `level` counter is never reset between passes -- it only increases, so stale
# `root_stamp` entries from an earlier map always compare as older and the O(N) reset
# can be skipped. `active` is cleared at the end of each pass, touching only the elements
# that pass actually set.
# --------------------------------------------------------------------------------------
mutable struct Workspace
    N::Int
    level::Int32
    ord::Vector{KeyIndex}  # positive part of the current map, sorted ascending by value
    ord_tmp::Vector{KeyIndex}  # the counting sort's scatter target
    cnt::Vector{Int32}    # the counting sort's exponent histogram
    parent::Vector{Int32}  # union-find forest, per element
    sz::Vector{Int32}      # union by size
    head::Vector{Int32}    # per root: first not-yet-died max-tree node
    tail::Vector{Int32}    # per root: last  not-yet-died max-tree node
    nd_next::Vector{Int32} # alive list, per node
    elem_node::Vector{Int32}    # leaf node of each element
    root_stamp::Vector{Int32}   # level at which each root last got a node
    active::Vector{UInt8}       # element is above the current height
    nd_parent::Vector{Int32}    # max-tree nodes, in creation order
    nd_size::Vector{Int32}
    nd_birth::Vector{Float64}   # birth^(H+1): the integral is accumulated in double
    nd_death::Vector{Float64}
    cum::Vector{Float64}
    sizeE::Vector{Float64}      # k^E for k = 1:N, so the sweep never calls `pow` on a size
end

function Workspace(N::Int, E::Float64)
    i32 = x -> Vector{Int32}(undef, x)
    w = Workspace(
        N,
        Int32(0),
        Vector{KeyIndex}(undef, N),
        Vector{KeyIndex}(undef, N),
        Vector{Int32}(undef, _MAX_EXPSPAN + 1),
        i32(N),                                      # parent
        i32(N),                                      # sz
        i32(N),                                      # head
        i32(N),                                      # tail
        i32(N),                                      # nd_next
        i32(N),                                      # elem_node
        i32(N),                                      # root_stamp
        Vector{UInt8}(undef, N),                     # active
        i32(N),                                      # nd_parent
        i32(N),                                      # nd_size
        Vector{Float64}(undef, N),                   # nd_birth
        Vector{Float64}(undef, N),                   # nd_death
        Vector{Float64}(undef, N),                   # cum
        Vector{Float64}(undef, N),                   # sizeE
    )
    fill!(w.active, 0x00)
    fill!(w.root_stamp, Int32(-1))
    for k = 1:N
        w.sizeE[k] = Float64(k)^E
    end
    return w
end

# Inclusive column range for worker `i` of `parts`: the first `r` workers get one
# extra column, and every worker gets a non-empty range for any `i <= parts <= n`.
function _column_chunk(n::Int, parts::Int, i::Int)
    q, r = divrem(n, parts)
    first = (i - 1) * q + min(i - 1, r) + 1
    return first, first + q - 1 + (i <= r ? 1 : 0)
end

# One map, both signs. `out` is written (it is freshly zeroed by the caller).
function _tfce_map!(
    out,
    col,
    adj::CSRAdjacency,
    w::Workspace,
    E::Float64,
    H::Float64,
    two_sided::Bool,
)
    Hp1 = H + 1
    _tfce_pass!(out, col, adj, w, Hp1, 1.0)
    two_sided && _tfce_pass!(out, col, adj, w, Hp1, -1.0)
    return out
end

# --------------------------------------------------------------------------------------
# Sort. Ordering the positive elements of a map is the sweep's first and, for a few
# hundred channels, most expensive step. Every value reaching here is strictly positive,
# and for positive IEEE-754 floats the bit pattern read as a `UInt64` increases
# monotonically with the value, so the key needs no transformation at all: it is the
# value. The channel index rides along in the second half of the record, so ordering
# never has to chase an indirection.
#
# The index cannot share the key's word (64 + 32 bits), hence the 16-byte record -- the
# same layout as the `VI` struct the C core sorts.
#
# The key being the raw bit pattern is what makes a counting sort possible. The pattern
# is `exponent << 52 | mantissa`, so *all keys sharing an exponent are contiguous in
# sorted order*: bucketing on the exponent is an exact partition, not a quantisation, and
# only the handful of keys inside a bucket still need comparing. Exponent spans within a
# single map are tiny -- measured on the benchmark data, median 5-8 and max 22 -- so the
# histogram is cheap to clear and buckets hold a few elements each. Above `_BUCKET_MIN`
# that beats both insertion sort and any comparison sort, and unlike insertion sort it
# barely grows with `n` (0.26 ms per 625 maps at m=32, 0.27 ms at m=64, 0.38 ms at
# m=128, against 0.64/0.64/1.30 for insertion sort).
#
# Below `_BUCKET_MIN` a plain insertion sort wins: the counting sort's fixed costs
# (clearing the histogram, the extra counting pass) do not pay for themselves when there
# are only a few keys. Measured against `Base.Sort.QuickSort` on maps drawn like the
# benchmark data, insertion sort is already 1.03x at m=16, 1.24x at m=32 and 1.14x at
# m=64, and the counting sort is 1.21x/1.42x at those sizes. A wide exponent span (only
# reachable for data spanning many binades) falls back to `QuickSort`, which is the one
# Base algorithm that does not allocate: `sort!`'s default resolves to a `ScratchQuickSort`
# that mallocs a scratch buffer on every call, ~1.7x slower.
#
# Equal values are not a problem for the sweep: it consumes a whole tie group as one
# level, and both the resulting partition and every element's score are independent of
# the order the group is merged in.
# --------------------------------------------------------------------------------------
const _BUCKET_MIN = 40
const _MAX_EXPSPAN = 24

function _sort_positive!(ord, ord_tmp, cnt, n::Int)
    if n <= _BUCKET_MIN
        _insertion_sort!(ord, 1, n)
    elseif !_bucket_sort!(ord, ord_tmp, cnt, n, _MAX_EXPSPAN)
        sort!(ord, 1, n, Base.Sort.QuickSort, Base.Order.Forward)
    end
    return n
end

# Exponent field of a positive double (bits 52..62).
@inline _expof(k::UInt64) = Int((k >> 52) & 0x7ff)

"""
    _bucket_sort!(ord, ord_tmp, cnt, n, maxspan) -> Bool

Counting sort of `ord[1:n]` by key exponent into `ord_tmp`, then insertion sort inside
each bucket, then copy back. Returns `false` without touching `ord` if the exponent span
exceeds `maxspan`, leaving the caller to fall back to a comparison sort. `cnt` must hold
at least `maxspan + 1` entries.
"""
function _bucket_sort!(ord, ord_tmp, cnt, n::Int, maxspan::Int)
    @inbounds begin
        emin = _expof(ord[1][1])
        emax = emin
        for i = 1:n
            e = _expof(ord[i][1])
            e < emin && (emin = e)
            e > emax && (emax = e)
        end
        span = emax - emin
        span > maxspan && return false
        nb = span + 1
        for k = 1:nb
            cnt[k] = 0
        end
        for i = 1:n                                       # histogram
            cnt[_expof(ord[i][1])-emin+1] += Int32(1)
        end
        s = Int32(1)
        for k = 1:nb                                      # exclusive prefix sum:
            c = cnt[k]                                    # cnt[k] becomes bucket k's
            cnt[k] = s                                    # first slot
            s += c
        end
        for i = 1:n                                       # scatter, bucketed by exponent
            b = _expof(ord[i][1]) - emin + 1
            j = cnt[b]
            ord_tmp[j] = ord[i]
            cnt[b] = j + Int32(1)
        end
        lo = 1                                            # after the scatter cnt[k] is one
        for k = 1:nb                                      # past the end of bucket k
            hi = Int(cnt[k]) - 1
            _insertion_sort!(ord_tmp, lo, hi)
            lo = hi + 1
        end
    end
    copyto!(ord, 1, ord_tmp, 1, n)
    return true
end

# Insertion sort of `ord[lo:hi]` by key. `>` compares keys only, which leaves equal
# values in the order they were gathered in.
@inline function _insertion_sort!(ord::Vector{KeyIndex}, lo::Int, hi::Int)
    @inbounds for i = (lo+1):hi
        x = ord[i]
        j = i - 1
        while j >= lo && ord[j][1] > x[1]
            ord[j+1] = ord[j]
            j -= 1
        end
        ord[j+1] = x
    end
    return ord
end

# --------------------------------------------------------------------------------------
# One signed pass over the positive part of `col`, accumulating TFCE scores into `out`
# with weight `sgn` (which is exactly ±1, so scaling the map in and the scores out is
# exact).
# --------------------------------------------------------------------------------------
function _tfce_pass!(out, col, adj::CSRAdjacency, w::Workspace, Hp1::Float64, sgn::Float64)
    N = w.N
    ord = w.ord
    n = 0                                                     # gather the positive part
    # Branchless: `x > 0` is a coin flip on noise-like data, so a conditional store here
    # mispredicts about half the time. Writing the record unconditionally and folding the
    # predicate into the counter is ~12% faster overall; the slot written when the test
    # fails is at most `ord[N]` and is never read (only `ord[1:n]` is used below).
    @inbounds for i = 1:N
        x = Float64(col[i]) * sgn
        ord[n+1] = (reinterpret(UInt64, x), Int32(i))
        n += ifelse(x > 0.0, 1, 0)
    end
    n == 0 && return out
    _sort_positive!(ord, w.ord_tmp, w.cnt, n)

    if w.level > typemax(Int32) - Int32(N)     # keep `root_stamp` comparisons valid
        fill!(w.root_stamp, Int32(-1))
        w.level = Int32(0)
    end
    level = w.level
    parent = w.parent
    sz = w.sz
    head = w.head
    tail = w.tail
    nd_next = w.nd_next
    active = w.active
    root_stamp = w.root_stamp
    elem_node = w.elem_node
    nd_parent = w.nd_parent
    nd_size = w.nd_size
    nd_birth = w.nd_birth
    nd_death = w.nd_death
    ptr = adj.ptr
    idx = adj.idx
    nNodes = Int32(0)

    p = n                                                     # walk the keys from the top
    while p >= 1
        key = ord[p][1]
        hpow = reinterpret(Float64, key)^Hp1        # shared by every node of this
        level += Int32(1)                           # level: `pow` once per level
        q = p
        @inbounds while q >= 1 && ord[q][1] == key  # ties share one level
            q -= 1
        end

        @inbounds for t = (q+1):p                             # activate as singletons
            u = ord[t][2]
            active[u] = 0x01
            parent[u] = u
            sz[u] = Int32(1)
            head[u] = NONODE
            tail[u] = NONODE
        end
        @inbounds for t = (q+1):p                             # merge with active neighbours
            u = ord[t][2]
            ru = _findroot!(parent, u)          # u's root: the same for every neighbour
            for k = (Int(ptr[u])+1):Int(ptr[u+1])
                v = idx[k]
                # `parent[v] == ru` already proves v's root is ru, i.e. the merge
                # would be a no-op, so the find can be skipped; the same load is
                # handed to the find that does run.
                if active[v] != 0x00 && parent[v] != ru
                    ru = _merge!(parent, sz, head, tail, nd_next, ru, v, parent[v])
                end
            end
        end
        @inbounds for t = (q+1):p                # one fresh node per changed component
            u = ord[t][2]
            r = _findroot!(parent, u)
            if root_stamp[r] != level
                nNodes += Int32(1)
                node = nNodes
                root_stamp[r] = level
                nd_parent[node] = Int32(0)
                nd_size[node] = sz[r]
                nd_birth[node] = hpow
                nd_death[node] = 0.0
                o = head[r]                      # every old component of r dies here
                while o != NONODE
                    nd_parent[o] = node
                    nd_death[o] = hpow
                    o = nd_next[o]
                end
                head[r] = node
                tail[r] = node
                nd_next[node] = NONODE
            end
            elem_node[u] = head[r]
        end
        p = q
    end
    w.level = level

    cum = w.cum
    sizeE = w.sizeE
    @inbounds for i = nNodes:-1:Int32(1)        # a node's parent always has a larger
        c = sizeE[nd_size[i]] * (nd_birth[i] - nd_death[i]) / Hp1   # index, so one
        par = nd_parent[i]                                      # backward pass suffices
        cum[i] = par == 0 ? c : c + cum[par]
    end
    @inbounds for t = 1:n
        u = ord[t][2]
        out[u] += sgn * cum[elem_node[u]]
        active[u] = 0x00                        # clear only what this pass touched
    end
    return out
end

# Merge the component of `v` into the one rooted at `ra` (union by size, splicing the
# merged node lists along in O(1)) and return the surviving root. Taking the caller's
# current root instead of an element saves a `findroot` per active neighbour, and
# `pv = parent[v]` -- which the caller has just loaded for its activity test -- saves a
# second.
@inline function _merge!(parent, sz, head, tail, nd_next, ra::Int32, v::Int32, pv::Int32)
    rb = _findroot!(parent, v, pv)
    ra == rb && return ra
    if sz[ra] < sz[rb]
        ra, rb = rb, ra
    end
    parent[rb] = ra
    sz[ra] += sz[rb]
    hb = head[rb]
    if hb != NONODE
        if head[ra] == NONODE
            head[ra] = hb
            tail[ra] = tail[rb]
        else
            nd_next[tail[ra]] = hb
            tail[ra] = tail[rb]
        end
        head[rb] = NONODE
        tail[rb] = NONODE
    end
    return ra
end

# Union-find find with path halving. `px` must be `parent[x]`: the caller's activity test
# has usually loaded it already, and handing it over saves a redundant load. Starting the
# walk at the parent rather than at `x` reaches the same root by the same steps, minus the
# first hop, and leaves exactly the same forest the plain two-argument form would.
@inline function _findroot!(parent, x::Int32, px::Int32)
    @inbounds while px != x
        ppx = parent[px]                 # grandparent
        parent[x] = ppx                  # halve: point x straight at it
        x = ppx
        px = parent[x]                   # invariant: px == parent[x]
    end
    return x
end

@inline function _findroot!(parent, x::Int32)
    @inbounds return _findroot!(parent, x, parent[x])
end

end # module
