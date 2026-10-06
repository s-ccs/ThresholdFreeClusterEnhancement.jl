# --------------------------------------------------------------------------------------
# Workspace: every per-pass array, allocated once per thread and reused across all maps.
# The `level` counter is never reset between passes -- it only increases, so stale
# `root_stamp` entries from an earlier map always compare as older and the O(N) reset
# can be skipped. `parent` doubles as the activity flag: it is zero exactly for elements
# not yet activated in the current pass, and the sweep clears it again on its way out.
# --------------------------------------------------------------------------------------
mutable struct Workspace
    N::Int
    sortalg::Symbol
    level::Int
    ord::Vector{Tuple{Float64,Int}}    # positive part of the current map, sorted ascending
    ord_tmp::Vector{Tuple{Float64,Int}}    # the bucket sort's scatter target (:bucket only)
    cnt::Vector{Int}    # the bucket sort's exponent histogram (:bucket only)
    parent::Vector{Int}    # union-find forest, per element; 0 = not yet active
    sz::Vector{Int}    # union by size
    head::Vector{Int}    # per root: first not-yet-died max-tree node
    tail::Vector{Int}    # per root: last  not-yet-died max-tree node
    nd_next::Vector{Int}    # alive list, per node
    elem_node::Vector{Int}    # leaf node of each element
    root_stamp::Vector{Int}    # level at which each root last got a node
    nd_parent::Vector{Int}    # max-tree nodes, in creation order
    nd_size::Vector{Int}
    nd_birth::Vector{Float64}    # birth^(H+1): the integral is accumulated in double
    nd_death::Vector{Float64}
    cum::Vector{Float64}
    sizeE::Vector{Float64}    # k^E for k = 1:N, so the sweep never calls `pow` on a size
end

function Workspace(N::Int, E::Float64, sortalg::Symbol = :quick)
    bucket = sortalg === :bucket
    w = Workspace(
        N,
        sortalg,
        0,
        Vector{Tuple{Float64,Int}}(undef, N),
        bucket ? Vector{Tuple{Float64,Int}}(undef, N) : Tuple{Float64,Int}[],
        bucket ? Vector{Int}(undef, _MAX_EXPSPAN + 1) : Int[],
        zeros(Int, N),                               # parent
        Vector{Int}(undef, N),                       # sz
        Vector{Int}(undef, N),                       # head
        Vector{Int}(undef, N),                       # tail
        Vector{Int}(undef, N),                       # nd_next
        Vector{Int}(undef, N),                       # elem_node
        Vector{Int}(undef, N),                       # root_stamp
        Vector{Int}(undef, N),                       # nd_parent
        Vector{Int}(undef, N),                       # nd_size
        Vector{Float64}(undef, N),                   # nd_birth
        Vector{Float64}(undef, N),                   # nd_death
        Vector{Float64}(undef, N),                   # cum
        Vector{Float64}(undef, N),                   # sizeE
    )
    fill!(w.root_stamp, -1)
    for k = 1:N
        w.sizeE[k] = Float64(k)^E
    end
    return w
end
