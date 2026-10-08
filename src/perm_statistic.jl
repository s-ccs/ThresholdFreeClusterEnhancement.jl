# Permutation layer: a one-sample sign-flip test of a TFCE score map, built on
# PermutationTests.jl. `TFCEOneSampT` is a custom `PermutationTests.Statistic`;
# the permutation mechanism is the package's `OneSampStatistic` (`StudentT_1S`)
# sign-flip scheme, the same one its `_permMcTest!` driver uses. Because that
# driver only accumulates the maximum statistic, the driver here additionally
# accumulates the largest-K values and the exact exceedance count of each
# element's own null (the inputs the tail fits in tails.jl need).

"""
    TFCEOneSampT

A custom `PermutationTests.Statistic` for the one-sample sign-flip test
of a TFCE score map (see [`permutation_test`](@ref)).

The "elements" are the (channel, time) pairs of the score map. For each element
the one-sample *t*-statistic across subjects is the height, and TFCE over the
channel neighbourhood is the spatial statistic: the test statistic of an element
is the TFCE score at that (channel, time). The permutation mechanism is the
`OneSampStatistic` (`StudentT_1S`) sign-flip scheme of PermutationTests.jl: each
permutation multiplies every element's subject values by the same fresh random
sign vector.
"""
struct TFCEOneSampT <: PermutationTests.Statistic end

"""
    onesamp_t(y)

The one-sample *t*-statistic of a column of subject values `y`:
`mean(y) / (std(y; corrected = true) / sqrt(length(y)))`. This is what
`PermutationTests.jl`'s `StudentT_1S` computes, and the same one the reference
toolbox's `PermutedGLM` computes for a one-sample design. Returns `0.0` when
`length(y) < 2` or the sample standard deviation is exactly zero (the statistic
is then undefined).
"""
function onesamp_t(y::AbstractVector{<:Real})
    n = length(y)
    n >= 2 || return 0.0
    m = mean(y)
    s = std(y; mean = m, corrected = true)
    s == 0.0 && return 0.0
    return m * sqrt(n) / s
end

"""
    tfce_score_map(Y; adj, Nchan, Ntimes, E = 0.5, H = 2.0, two_sided = true, signs = nothing)

TFCE score map (channels × times, `Float32`) from per-element subject values.

`Y` is a vector of `Nchan * Ntimes` vectors, each the (already sign-flipped,
unless `signs` is given) subject values of one (channel, time) element in
column-major (channel fastest) order. The one-sample `t` of each element is
computed first, and TFCE is then applied over channels, reusing [`tfce`](@ref)
unmodified. If `signs` (a length-`n` vector of ±1) is given it is applied to
the elements first, so `Y` may hold the original data.
"""
function tfce_score_map(
    Y;
    adj,
    Nchan,
    Ntimes,
    E = 0.5,
    H = 2.0,
    two_sided = true,
    signs = nothing,
)
    M = length(Y)
    tmap = Vector{Float64}(undef, M)
    if isnothing(signs)
        @inbounds for i = 1:M
            tmap[i] = onesamp_t(Y[i])
        end
    else
        @inbounds for i = 1:M
            tmap[i] = onesamp_t(Y[i] .* signs)
        end
    end
    return tfce(reshape(tmap, Nchan, Ntimes), adj; E = E, H = H, two_sided = two_sided)
end

# --------------------------------------------------------------------------------------
# testStatistic methods for TFCEOneSampT (PermutationTests.jl extension point).
#
# The conventions mirror the package's own `statistic` methods for
# `StudentT_1S`:
#   * Monte Carlo: `𝐱` is the membership vector (unused), `𝐘` holds the already
#     sign-flipped data, and the whole score map is returned.
#   * Exact: `𝐱` is a tuple holding the sign pattern (as
#     `genPerms(::OneSampStatistic)` yields them) and `𝐘` holds the original
#     data, applied through `signs`. With an index `i` the `i`th element of the
#     score map is returned (METHOD 2 of the package API); without one the whole
#     map is returned. Computing one element recomputes the whole map, as it
#     must: TFCE couples the channels.
# --------------------------------------------------------------------------------------
PermutationTests.testStatistic(
    𝐱,
    𝐘,
    stat::TFCEOneSampT;
    adj,
    Nchan,
    Ntimes,
    E = 0.5,
    H = 2.0,
    two_sided = true,
    cpcd = nothing,
    kwargs...,
) = tfce_score_map(
    𝐘;
    adj = adj,
    Nchan = Nchan,
    Ntimes = Ntimes,
    E = E,
    H = H,
    two_sided = two_sided,
)

PermutationTests.testStatistic(
    𝐱::Tuple,
    𝐘,
    stat::TFCEOneSampT;
    adj,
    Nchan,
    Ntimes,
    E = 0.5,
    H = 2.0,
    two_sided = true,
    cpcd = nothing,
    kwargs...,
) = tfce_score_map(
    𝐘;
    adj = adj,
    Nchan = Nchan,
    Ntimes = Ntimes,
    E = E,
    H = H,
    two_sided = two_sided,
    signs = 𝐱,
)

PermutationTests.testStatistic(
    𝐱::Tuple,
    𝐘,
    i::Int,
    stat::TFCEOneSampT;
    adj,
    Nchan,
    Ntimes,
    E = 0.5,
    H = 2.0,
    two_sided = true,
    cpcd = nothing,
    kwargs...,
) = tfce_score_map(
    𝐘;
    adj = adj,
    Nchan = Nchan,
    Ntimes = Ntimes,
    E = E,
    H = H,
    two_sided = two_sided,
    signs = 𝐱,
)[i]

# --------------------------------------------------------------------------------------
# Null accumulator: the maximum statistic per permutation (the FWE input) plus,
# per element, the largest `K` permuted values and the exact exceedance count
# over ALL permutations (the uncorrected inputs).
# --------------------------------------------------------------------------------------
mutable struct NullAccumulator
    null_max::Vector{Float64}       # max |score| per permutation, (n_perm,)
    tails::Vector{Vector{Float64}}  # per element: the largest values, sorted descending
    cnt::Vector{Int32}              # per element: how often |perm| >= |obs|
    K::Int
    j::Int                          # permutations accumulated so far

    function NullAccumulator(M::Int, n_perm::Int, K::Int)
        tails = [Vector{Float64}(undef, 0) for _ = 1:M]
        new(Vector{Float64}(undef, n_perm), tails, zeros(Int32, M), K, 0)
    end
end

function _insert_klargest!(buf::Vector{Float64}, v::Float64, K::Int)
    if length(buf) < K
        push!(buf, v)
        sort!(buf, rev = true)
        return
    end
    if v > buf[end]                 # buf is sorted descending: buf[end] is the smallest
        buf[end] = v
        i = length(buf)
        while i > 1 && buf[i-1] < v
            buf[i], buf[i-1] = buf[i-1], buf[i]
            i -= 1
        end
    end
end

function accumulate!(acc::NullAccumulator, s::Vector{Float64}, obs::Vector{Float64})
    acc.j += 1
    acc.null_max[acc.j] = maximum(s)
    @inbounds for i in eachindex(s)
        acc.cnt[i] += (s[i] >= obs[i])
        _insert_klargest!(acc.tails[i], s[i], acc.K)
    end
    return acc
end

# The TFCE score map for one permutation of `Y` (already sign-flipped), as a
# flat vector of magnitudes, into preallocated scratch.
function _permuted_magnitudes!(scratch, Y, x, stat, adj, Nchan, Ntimes, E, H, two_sided)
    score = PermutationTests.testStatistic(
        x,
        Y,
        stat;
        adj = adj,
        Nchan = Nchan,
        Ntimes = Ntimes,
        E = E,
        H = H,
        two_sided = two_sided,
    )
    M = length(scratch)
    flat = reshape(score, :)
    @inbounds for i = 1:M
        scratch[i] = abs(Float64(flat[i]))
    end
    return scratch
end

"""
    _run_null!(Y, x, stat, acc, s, obs, adj, Nchan, Ntimes, E, H, two_sided; nperm, rng)

Monte-Carlo null of the TFCE score map: `nperm` fresh sign-flip permutations,
each time accumulating the maximum statistic, the per-element tails and the
per-element exceedance counts. `Y` is modified in place (cumulative sign
flips; each permutation still applies a fresh uniform sign vector).
"""
function _run_null!(
    Y,
    x,
    stat,
    acc::NullAccumulator,
    s,
    obs,
    adj,
    Nchan,
    Ntimes,
    E,
    H,
    two_sided;
    nperm,
    rng,
)
    @inbounds for j = 1:nperm
        # the package's sign-flip scheme: fresh random signs, applied to `Y`
        PermutationTests._randperm_multComp!(x, Y, PermutationTests.StudentT_1S(), rng)
        _permuted_magnitudes!(s, Y, x, stat, adj, Nchan, Ntimes, E, H, two_sided)
        accumulate!(acc, s, obs)
    end
    return acc
end

"""
    _run_null_exact!(Yorig, x, stat, acc, s, obs, adj, Nchan, Ntimes, E, H, two_sided; n)

Exact null of the TFCE score map: all `2^n` sign patterns, enumerated by
`PermutationTests.genPerms` (the systematic permutations of the `OneSampStatistic`
scheme) and applied to the original data `Yorig`, which is never modified.
"""
function _run_null_exact!(
    Yorig,
    x,
    stat,
    acc::NullAccumulator,
    s,
    obs,
    adj,
    Nchan,
    Ntimes,
    E,
    H,
    two_sided;
    n,
)
    design = PermutationTests.Balanced()
    direction = PermutationTests.Both()
    for p in
        PermutationTests.genPerms(PermutationTests.StudentT_1S(), x, n, direction, design)
        _permuted_magnitudes_exact!(s, Yorig, p, stat, adj, Nchan, Ntimes, E, H, two_sided)
        accumulate!(acc, s, obs)
    end
    return acc
end

function _permuted_magnitudes_exact!(s, Yorig, p, stat, adj, Nchan, Ntimes, E, H, two_sided)
    # p is the sign pattern tuple (the exact-test convention: apply it to the
    # original data, never to already-flipped data)
    score = PermutationTests.testStatistic(
        p,
        Yorig,
        stat;
        adj = adj,
        Nchan = Nchan,
        Ntimes = Ntimes,
        E = E,
        H = H,
        two_sided = two_sided,
    )
    M = length(s)
    flat = reshape(score, :)
    @inbounds for i = 1:M
        s[i] = abs(Float64(flat[i]))
    end
    return s
end
