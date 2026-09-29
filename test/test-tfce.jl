@testsnippet TFCEHelpers begin
    using Random
    using LinearAlgebra
    using ThresholdFreeClusterEnhancement

    # Deliberately naive reference: step thresholds on a grid and relabel connected
    # components (BFS) at every step. First-order accurate in the step size; used only
    # to validate the exact max-tree result.
    function naive_tfce_pass(d, adj, E, H, n_steps)
        N = length(d)
        out = zeros(N)
        hi = maximum(d)
        hi <= 0 && return out
        dh = hi / n_steps
        for k = 1:n_steps
            h = k * dh
            act = d .>= h
            comp = zeros(Int, N)
            ncomp = 0
            for s = 1:N
                (act[s] && comp[s] == 0) || continue
                ncomp += 1
                comp[s] = ncomp
                stack = [s]
                while !isempty(stack)
                    u = pop!(stack)
                    for v in adj[u]
                        if act[v] && comp[v] == 0
                            comp[v] = ncomp
                            push!(stack, v)
                        end
                    end
                end
            end
            sizes = zeros(Int, ncomp)
            for u = 1:N
                comp[u] != 0 && (sizes[comp[u]] += 1)
            end
            for u = 1:N
                comp[u] == 0 && continue
                out[u] += sizes[comp[u]]^E * h^H * dh
            end
        end
        return out
    end

    function naive_tfce(d, adj; E, H, n_steps, two_sided)
        out = naive_tfce_pass(d, adj, E, H, n_steps)
        two_sided && (out .-= naive_tfce_pass(.-d, adj, E, H, n_steps))
        return out
    end

    "Line (path) adjacency for `C` channels: i is adjacent to i±1."
    path_adjacency(C) = diagm(1 => ones(C - 1), -1 => ones(C - 1))

    "Random 2-D channel layout; channels closer than `radius` are neighbours."
    function random_layout_adjacency(rng, C; radius = 0.5)
        pos = [(rand(rng), rand(rng)) for _ = 1:C]
        A = zeros(Bool, C, C)
        for i = 1:C, j = 1:C
            A[i, j] = i != j && norm(pos[i] .- pos[j]) < radius
        end
        return A
    end
end

@testitem "hand-computed tiny maps" tags = [:unit, :validation] setup = [TFCEHelpers] begin
    # Line 1-2-3-4, values [0, 3, 1, 2], E=1, H=0: component history is {2} on [1,3],
    # {4} on [1,2], {2,3,4} on [0,1] -> TFCE = [0, 5, 3, 4] (hand-derived in PLAN.md).
    line = path_adjacency(4)
    @test tfce([0.0, 3.0, 1.0, 2.0], line; E = 1, H = 0) == Float32[0, 5, 3, 4]

    # Two connected channels merge: [4, 2]; two disconnected ones stay separate: [3, 1].
    @test tfce([3.0, 1.0], Bool[0 1; 1 0]; E = 1, H = 0) == Float32[4, 2]
    @test tfce([3.0, 1.0], zeros(2, 2); E = 1, H = 0) == Float32[3, 1]
end

@testitem "E = 0 makes TFCE extent-free" tags = [:unit, :validation] setup = [TFCEHelpers] begin
    # With E = 0 every node contributes 1·(b^(H+1) − d^(H+1))/(H+1); the intervals on a
    # root->leaf path tile [0, t_v], so TFCE(v) = t_v^(H+1)/(H+1) for any adjacency.
    rng = MersenneTwister(7)
    A = random_layout_adjacency(rng, 12; radius = 0.6)
    d = 3 .* randn(rng, 12)
    out = tfce(d, A; E = 0.0, H = 2.0, two_sided = true)
    @test out ≈ sign.(d) .* abs.(d) .^ 3 ./ 3 atol = 1e-5
end

@testitem "all-zero map stays zero" tags = [:unit, :validation] setup = [TFCEHelpers] begin
    A = path_adjacency(4)
    @test tfce(zeros(4, 3), A) == zeros(4, 3)
    @test tfce(zeros(4, 3, 2), A) == zeros(4, 3, 2)
end

@testitem "single spike: adjacency respected" tags = [:unit, :validation] setup =
    [TFCEHelpers] begin
    A = path_adjacency(5)                       # 1-2-3-4-5
    d = zeros(5)
    d[3] = 5.0
    out = tfce(d, A; two_sided = false)
    @test out[3] > 0
    @test all(iszero, out[[1, 2, 4, 5]])

    d[2] = 1.0                                  # neighbour becomes active too
    out = tfce(d, A; two_sided = false)
    @test out[3] > 0 && out[2] > 0
    @test all(iszero, out[[1, 4, 5]])           # inactive channels never light up
end

@testitem "two-sided behaviour" tags = [:unit, :validation] setup = [TFCEHelpers] begin
    A = path_adjacency(5)
    d = zeros(5)
    d[3] = -5.0
    @test tfce(d, A; two_sided = true)[3] < 0
    @test all(iszero, tfce(d, A; two_sided = false))

    pos = abs.(d) .+ 1.0                        # strictly positive map
    @test tfce(pos, A; two_sided = true) == tfce(pos, A; two_sided = false)
end

@testitem "no cross-sign leakage" tags = [:unit, :validation] setup = [TFCEHelpers] begin
    A = path_adjacency(5)
    d = [3.0, 1.0, 0.0, -1.0, -3.0]
    @test tfce(d, A; two_sided = false) == tfce(max.(d, 0), A; two_sided = false)
end

@testitem "columns and subjects are independent" tags = [:unit, :validation] setup =
    [TFCEHelpers] begin
    rng = MersenneTwister(11)
    A = random_layout_adjacency(rng, 10; radius = 0.6)
    d = 3 .* randn(rng, 10, 4)

    dup = tfce(hcat(d[:, 1], d[:, 1]), A)       # duplicate time columns
    @test dup[:, 1] == dup[:, 2]

    data3 = cat(d, 2 .* d; dims = 3)            # (channels, times, subjects)
    out3 = tfce(data3, A)
    @test out3 == cat([tfce(data3[:, :, s], A) for s = 1:2]...; dims = 3)
end

@testitem "richer adjacency merges more" tags = [:unit, :validation] setup = [TFCEHelpers] begin
    d = [2.0, 0.0, 2.0, 1.0]
    path = path_adjacency(4)                    # 1-2-3-4
    rich = path_adjacency(4)
    rich[1, 3] = rich[3, 1] = 1.0               # extra 1-3 link
    @test sum(tfce(d, rich; two_sided = false)) > sum(tfce(d, path; two_sided = false))
end

@testitem "exactness: converges onto naive dh-stepping reference" tags =
    [:unit, :validation] setup = [TFCEHelpers] begin
    rng = MersenneTwister(42)
    C = 16
    A = random_layout_adjacency(rng, C; radius = 0.35)
    adj = ThresholdFreeClusterEnhancement.adjacency_lists(A)
    d = round.(3 .* randn(rng, C); digits = 1)  # rounding creates ties on purpose

    for (E, H, ts) in [(0.5, 2.0, true), (1.0, 2.0, false)]
        exact = tfce(d, A; E, H, two_sided = ts)
        errs = [
            maximum(abs, naive_tfce(d, adj; E, H, n_steps = ns, two_sided = ts) .- exact) for ns in (50, 100, 200, 400)
        ]
        @test all(errs[i+1] < errs[i] for i = 1:3)
        slope = (log(errs[1]) - log(errs[4])) / (log(50) - log(400))
        @test slope ≈ -1 atol = 0.25            # first-order convergence
        @test errs[4] < 0.1 * maximum(abs, exact)
    end
end
