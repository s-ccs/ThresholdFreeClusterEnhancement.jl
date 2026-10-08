# Benchmark the Julia sign-flip permutation test against the reference Python
# workflow (see perm_benchmark.py for the documented recipe it times, and
# README.md for the protocol).
#
# Run `generate_data.py` and `perm_benchmark.py` first (see README.md); this
# script loads the same bit-identical data, times `permutation_test` on it,
# and cross-checks the result against the outputs the Python side wrote.

using ThresholdFreeClusterEnhancement

const CASES = [("small", 32, 128, 10), ("medium", 64, 256, 20), ("large", 128, 512, 30)]

const DATA = joinpath(@__DIR__, "data")
const NPERM = 1000
const REPEATS = 3

function load_array(::Type{T}, dims, name) where {T}
    arr = Array{T}(undef, dims)
    open(joinpath(DATA, name), "r") do f
        read!(f, arr)
    end
    return arr
end

function main()
    for (name, C, T, S) in CASES
        data = load_array(Float64, (C, T, S), "$name.data.bin")
        adj = load_array(Float64, (C, C), "$name.adjacency.bin")

        permutation_test(data, adj; nperm = 20)  # warm-up
        times = [@elapsed permutation_test(data, adj; nperm = NPERM) for _ = 1:REPEATS]

        # cross-check against the outputs the Python side wrote
        obs_py = load_array(Float32, (C, T), "$name.perm.obs.python.bin")
        pfwe_py = load_array(Float64, (C * T,), "$name.perm.pfwe.python.bin")
        punc_py = load_array(Float64, (C * T,), "$name.perm.punc.python.bin")

        pfwe, punc = permutation_test(data, adj; nperm = NPERM)

        # observed score map: deterministic on both sides, float32 precision
        M = C * T
        Yorig = [Vector{Float64}(data[(i-1)%C+1, (i-1)÷C+1, :]) for i = 1:M]
        score0 = tfce_score_map(Yorig; adj = adj, Nchan = C, Ntimes = T)
        d_obs = maximum(abs, vec(score0) .- vec(obs_py))

        # p_fwe: same convention on both sides (maximum |score|, Gamma fit),
        # independent random draws, so this is Monte-Carlo noise
        d_pfwe = maximum(abs, vec(pfwe) .- pfwe_py)

        # p_unc: the reference's is sign-conditioned (one tail, uniform on
        # (0, 0.5]); Julia counts magnitudes on both sides (uniform on (0, 1]),
        # so under the symmetric null p_julia ≈ 2 * p_python. The two sides
        # drew different sign streams, so compare where both fits are
        # constrained: the mid-regime, where the factor of 2 stays well below 1.
        mask = (punc_py .>= 0.01) .& (2 .* punc_py .<= 0.8)
        d_punc = maximum(abs, vec(punc)[mask] .- 2 .* punc_py[mask])
        d_punc_raw = maximum(abs, vec(punc) .- punc_py)

        best = minimum(times) * 1000
        println(
            "$name: C=$C T=$T S=$S  nperm=$NPERM  best: $(round(best / 1000, digits=2)) s  " *
            "runs (s): $(join([round(t, digits=2) for t in times], ", "))  " *
            "max |Δobs|: $(round(d_obs, sigdigits=3))  " *
            "max |Δp_fwe|: $(round(d_pfwe, sigdigits=3))  " *
            "max |Δp_unc − 2·p_py|: $(round(d_punc, sigdigits=3))  " *
            "max |Δp_unc| (raw, different conventions): $(round(d_punc_raw, sigdigits=3))",
        )
    end
end

main()
