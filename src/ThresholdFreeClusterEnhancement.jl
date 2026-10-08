module ThresholdFreeClusterEnhancement

using LinearAlgebra
using Random
using StatsBase
using SpecialFunctions: gamma_inc
using PermutationTests

export tfce
export permutation_test, tfce_score_map, onesamp_t, TFCEOneSampT
export gamma_pvalue, pareto_pvalue, gpd_fit_pwm, gpd_sf

include("adjacency.jl")        # neighbour lists + flat CSR adjacency
include("workspace.jl")        # per-thread preallocated workspace
include("sort.jl")             # sorting: built-in by default, experimental `:bucket` opt-in
include("sweep.jl")            # the max-tree + union-find sweep over one map
include("tfce.jl")             # public entry point and threading
include("tails.jl")            # Winkler tail fits (Gamma for FWE, GPD for uncorrected)
include("perm_statistic.jl")   # custom Statistic + sign-flip null accumulator
include("permutation_test.jl") # public entry point: one-sample sign-flip TFCE test

end # module
