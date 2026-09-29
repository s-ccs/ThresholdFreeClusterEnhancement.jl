# Benchmark the barebone Julia TFCE against the reference Python/C implementation.
#
# Run `generate_data.py` and `benchmark.py` first (see README.md); this script
# loads the same bit-identical data, times `tfce` on it, and cross-checks the
# result against the output the Python side wrote.

using ThresholdFreeClusterEnhancement

const CASES = [("small", 32, 128, 10), ("medium", 64, 256, 20), ("large", 128, 512, 30)]

const DATA = joinpath(@__DIR__, "data")
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

        tfce(data, adj)  # warm-up
        times = [@elapsed tfce(data, adj) for _ = 1:REPEATS]

        # cross-check against the reference result, written by benchmark.py
        py = load_array(Float32, (C, T * S), "$name.out.python.bin")
        out = tfce(data, adj)
        diff = maximum(abs, reshape(out, C, T * S) .- py)

        best = minimum(times) * 1000
        println(
            "$name: channels=$C cols=$(T*S)  best: $(round(best, digits=2)) ms  " *
            "runs (ms): $(join([round(t * 1000, digits=2) for t in times], ", "))  " *
            "max |julia - python|: $(round(diff, sigdigits=3))",
        )
    end
end

main()
