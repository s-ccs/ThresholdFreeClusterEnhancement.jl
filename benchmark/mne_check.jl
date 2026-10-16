# Cross-check the Julia TFCE against MNE-Python's independent implementation
# (see mne_check.py for the MNE side and README.md for the protocol).
#
# Run `mne_check.py` first; this script loads the shared data, runs the Julia
# transform and the exact permutation test on it, and compares against the
# outputs the MNE side wrote.

using ThresholdFreeClusterEnhancement
using Statistics

const CASES = [("small", 32, 128, 10)]

const DATA = joinpath(@__DIR__, "data")

function load_array(::Type{T}, dims, name) where {T}
    arr = Array{T}(undef, dims)
    open(joinpath(DATA, name), "r") do f
        read!(f, arr)
    end
    return arr
end

# tied ranks (average ties), for the Spearman correlation
function _ranks(v)
    order = sortperm(v)
    sv = v[order]
    r = zeros(Float64, length(v))
    i = 1
    n = length(v)
    while i <= n
        j = i
        while j < n && sv[j+1] == sv[i]
            j += 1
        end
        avg = (i + j) / 2
        for k = i:j
            r[order[k]] = avg
        end
        i = j + 1
    end
    return r
end

spearman(x, y) = cor(_ranks(x), _ranks(y))

function main()
    fine = isfile(joinpath(DATA, "small.mne.obs.fine.bin"))
    for (name, C, T, S) in CASES
        data = load_array(Float64, (C, T, S), "$name.data.bin")
        adj = load_array(Float64, (C, C), "$name.adjacency.bin")

        # Julia observed score map and exact-test p-values; tfce_score_map is
        # signed (two-sided), so take magnitudes -- MNE's obs output is |scores|
        M = C * T
        Yorig = [Vector{Float64}(data[(i-1)%C+1, (i-1)÷C+1, :]) for i = 1:M]
        score0 =
            abs.(Float64.(vec(tfce_score_map(Yorig; adj = adj, Nchan = C, Ntimes = T))))
        pfwe, = permutation_test(data, adj; exact = true, method = :count)

        # MNE outputs
        tstat = vec(load_array(Float64, (C, T), "$name.mne.tstat.bin"))
        obs = vec(load_array(Float64, (C, T), "$name.mne.obs.bin"))
        p_mne = vec(load_array(Float64, (C, T), "$name.mne.p.bin"))

        # height map: the one-sample t across subjects, machine precision
        tmap = [
            begin
                d = @view data[c, t, :]
                σ = std(d)
                σ == 0 ? 0.0 : mean(d) / (σ / sqrt(S))
            end for c = 1:C, t = 1:T
        ]
        d_t = maximum(abs, vec(tmap) .- tstat)

        # observed score map: deterministic on both sides; the difference is
        # MNE's stepped (Riemann-sum) discretization, which shrinks with step
        d_obs = maximum(abs, score0 .- obs)
        rel = d_obs / maximum(abs, obs)
        rho_obs = spearman(score0, obs)

        # p_fwe: the exact test on both sides (same null, no Monte-Carlo
        # noise); the counting conventions differ by the observed-value slot,
        # so the floor of the difference is ~1/511 on the MNE side
        d_p = maximum(abs, Float64.(vec(pfwe)) .- p_mne)
        rho_p = spearman(Float64.(vec(pfwe)), p_mne)

        print(
            "$name: max|Δt|: $(round(d_t, sigdigits=3))  " *
            "max|ΔTFCE|: $(round(d_obs, sigdigits=3)) " *
            "(rel $(round(rel, sigdigits=3)), spearman $(round(rho_obs, digits=5)))  " *
            "max|Δp_fwe|: $(round(d_p, sigdigits=3)) " *
            "(floor ~$(round(1/511, sigdigits=2)), spearman $(round(rho_p, digits=5)))",
        )
        if fine
            obs_fine = vec(load_array(Float64, (C, T), "$name.mne.obs.fine.bin"))
            d_fine = maximum(abs, score0 .- obs_fine)
            rel_fine = d_fine / maximum(abs, obs_fine)
            print(
                "  |  fine step: max|ΔTFCE|: $(round(d_fine, sigdigits=3)) " *
                "(rel $(round(rel_fine, sigdigits=3)))",
            )
            if isfile(joinpath(DATA, "$name.mne.p.fine.bin"))
                p_fine = vec(load_array(Float64, (C, T), "$name.mne.p.fine.bin"))
                d_p_fine = maximum(abs, Float64.(vec(pfwe)) .- p_fine)
                print("  max|Δp_fwe|: $(round(d_p_fine, sigdigits=3))")
            end
        end
        println()
    end
end

main()
