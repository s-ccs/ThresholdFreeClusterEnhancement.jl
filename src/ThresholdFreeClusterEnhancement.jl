module ThresholdFreeClusterEnhancement

using LinearAlgebra
using StatsBase

export tfce

"""
    tfce(data, adjacency; E = 0.5, H = 2.0, two_sided = true) -> Array{Float32}

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

Returns scores with the same shape as `data`.
"""
function tfce(
    data::AbstractArray,
    adjacency::AbstractMatrix;
    E = 0.5,
    H = 2.0,
    two_sided = true,
)
    adj = adjacency_lists(adjacency)
    maps = reshape(data, size(data, 1), :)          # (channels, times·subjects)
    out = zeros(Float32, size(maps))
    for b in axes(maps, 2)
        d = Float64.(view(maps, :, b))
        _tfce_pass!(view(out, :, b), d, adj, E, H, +1)
        two_sided && _tfce_pass!(view(out, :, b), -d, adj, E, H, -1)
    end
    return reshape(out, size(data))
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

# One signed sweep over the positive part of `d`, accumulating TFCE scores into `out`.
function _tfce_pass!(out, d, adj, E, H, sgn)
    N = length(d)
    ord = filter!(i -> d[i] > 0, sortperm(d; rev = true))
    isempty(ord) && return out

    active = falses(N)                # element is above the current height
    parent = collect(1:N)             # union-find forest
    sz = ones(Int, N)                 # union by size
    root_stamp = fill(0, N)           # level at which each root last got a node
    elem_node = zeros(Int, N)         # leaf node of each element
    alive = [Int[] for _ = 1:N]      # per root: its not-yet-died tree nodes
    nd_parent = Int[]                 # max-tree nodes, appended in creation order
    nd_size = Int[]
    nd_birth = Float64[]
    nd_death = Float64[]
    level = 0

    heights, lengths = rle(d[ord])    # ties (equal values) form one level
    offset = 0
    for (h, len) in zip(heights, lengths)
        tie = @view ord[(offset+1):(offset+len)]
        offset += len
        level += 1

        for u in tie                  # 1. activate this level's elements
            active[u] = true
            parent[u] = u
            sz[u] = 1
        end
        for u in tie, v in adj[u]     # 2. merge with already-active neighbours
            if active[v]
                ra = _findroot!(parent, u)
                rb = _findroot!(parent, v)
                if ra != rb
                    if sz[ra] < sz[rb]
                        ra, rb = rb, ra
                    end
                    parent[rb] = ra
                    sz[ra] += sz[rb]
                    append!(alive[ra], alive[rb])
                    empty!(alive[rb])
                end
            end
        end
        for u in tie                  # 3. one max-tree node per changed component
            r = _findroot!(parent, u)
            if root_stamp[r] != level
                push!(nd_parent, 0)
                push!(nd_size, sz[r])
                push!(nd_birth, h)
                push!(nd_death, 0.0)
                node = length(nd_size)
                for old in alive[r]   # components merged into this one die here
                    nd_parent[old] = node
                    nd_death[old] = h
                end
                alive[r] = [node]
                root_stamp[r] = level
            end
            elem_node[u] = alive[_findroot!(parent, u)][1]
        end
    end

    Hp1 = H + 1                       # 4. root -> leaf accumulation (parents have larger
    cum = zeros(length(nd_size))      #    indices, so one backward pass suffices)
    for node = length(nd_size):-1:1
        c = nd_size[node]^E * (nd_birth[node]^Hp1 - nd_death[node]^Hp1) / Hp1
        cum[node] = c + (nd_parent[node] > 0 ? cum[nd_parent[node]] : 0.0)
    end
    for u in ord
        out[u] += sgn * cum[elem_node[u]]
    end
    return out
end

# Union-find find with path halving.
function _findroot!(parent, x)
    while parent[x] != x
        parent[x] = parent[parent[x]]
        x = parent[x]
    end
    return x
end

end # module
