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
    ptr::Vector{Int}     # row u occupies ptr[u]+1 : ptr[u+1]
    idx::Vector{Int}
end

function csr_adjacency(adjacency::AbstractMatrix)
    N = size(adjacency, 1)
    positions = findall(!iszero, adjacency)

    deg = zeros(Int, N)                       # count first, then fill (two O(nnz) sweeps)
    for pos in positions
        pos[1] == pos[2] && continue
        deg[pos[1]] += 1
    end
    ptr = Vector{Int}(undef, N + 1)
    ptr[1] = 0
    for u = 1:N
        ptr[u+1] = ptr[u] + deg[u]
    end
    idx = Vector{Int}(undef, ptr[N+1])
    fill = copy(ptr)
    for pos in positions
        u, v = pos[1], pos[2]
        u == v && continue
        idx[fill[u]+1] = v
        fill[u] += 1
    end
    return CSRAdjacency(ptr, idx)
end
