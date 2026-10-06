module ThresholdFreeClusterEnhancement

using LinearAlgebra
using StatsBase

export tfce

include("adjacency.jl")    # neighbour lists + flat CSR adjacency
include("workspace.jl")    # per-thread preallocated workspace
include("sort.jl")         # sorting: built-in by default, experimental `:bucket` opt-in
include("sweep.jl")        # the max-tree + union-find sweep over one map
include("tfce.jl")         # public entry point and threading

end # module
