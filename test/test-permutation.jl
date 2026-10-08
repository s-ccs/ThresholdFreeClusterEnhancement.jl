@testsnippet PermHelpers begin
    using Random
    using LinearAlgebra
    using Statistics: mean, std, median
    using ThresholdFreeClusterEnhancement
    using PermutationTests

    "Line (path) adjacency for `C` channels: i is adjacent to i±1."
    path_adjacency(C) = diagm(1 => ones(C - 1), -1 => ones(C - 1))

    # The per-element layout permutation_test uses: a vector of Nchan*Ntimes subject
    # value vectors, column-major (channel fastest) over a channels × times map.
    function Ylayout(data, Nchan, Ntimes)
        M = Nchan * Ntimes
        return [Vector{Float64}(data[(i-1)%Nchan+1, (i-1)÷Nchan+1, :]) for i = 1:M]
    end

    # Deliberately naive one-sample t: a different arithmetic path (dot product for
    # the sum of squares) than the package's mean/std, used as an independent refit.
    function naive_onesamp_t(y)
        n = length(y)
        m = sum(y) / n
        d = y .- m
        return m * sqrt(n) / sqrt(dot(d, d) / (n - 1))
    end

    # Honest naive refit of one sign-flip permutation: flip the raw data, recompute
    # the per-element t maps, then TFCE. Independent of the package's driver path.
    function naive_score_map(data, adjacency, signs; E = 0.5, H = 2.0)
        Nchan, Ntimes, n = size(data)
        tmap = zeros(Nchan, Ntimes)
        for c = 1:Nchan, t = 1:Ntimes
            tmap[c, t] = naive_onesamp_t(data[c, t, :] .* signs)
        end
        return tfce(tmap, adjacency; E = E, H = H, two_sided = true)
    end

    # The maximum of many independent statistics is Gumbel-like (test_tails.py).
    # One column per permutation, n_elem statistics per column.
    function gumbel_max(rng, n; n_elem = 200, shift = 0.0)
        return vec(maximum(-log.(-log.(rand(rng, n_elem, n))), dims = 1)) .+ shift
    end

    # Read n consecutive little-endian float64 values from an open binary stream.
    function readbin(f, n)
        b = Vector{UInt8}(undef, 8 * n)
        readbytes!(f, b)
        return reinterpret(Float64, b)
    end
end

@testitem "permuted statistic matches an honest naive refit" tags =
    [:permutation, :validation] setup = [PermHelpers] begin
    rng = MersenneTwister(21)
    Nchan, Ntimes, n = 8, 5, 8
    data = randn(rng, Nchan, Ntimes, n)
    data[3:5, 2:4, :] .+= 1.5  # planted effect
    adj = path_adjacency(Nchan)
    Y = Ylayout(data, Nchan, Ntimes)
    stat = TFCEOneSampT()
    x = PermutationTests.membership(PermutationTests.StudentT_1S(), n)
    signs = (1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0)

    # the exact convention: apply the sign pattern to the original data
    score = PermutationTests.testStatistic(
        signs,
        Y,
        stat;
        adj = adj,
        Nchan = Nchan,
        Ntimes = Ntimes,
    )
    naive = naive_score_map(data, adj, signs)

    # the t statistic itself, at floating-point precision (same element order)
    t_pkg = [onesamp_t(y .* signs) for y in Y]
    t_nai = [
        naive_onesamp_t(data[(i-1)%Nchan+1, (i-1)÷Nchan+1, :] .* signs) for
        i = 1:(Nchan*Ntimes)
    ]
    @test maximum(abs, t_pkg .- t_nai) < 1e-12 * maximum(abs, t_nai)

    # the score maps agree to Float32 precision (tfce's working precision)
    @test score ≈ naive rtol = 1e-6

    # the Monte-Carlo convention: the data arrives already sign-flipped
    Yf = [y .* signs for y in Y]
    score_mc = PermutationTests.testStatistic(
        x,
        Yf,
        stat;
        adj = adj,
        Nchan = Nchan,
        Ntimes = Ntimes,
    )
    @test score_mc ≈ naive rtol = 1e-6

    # METHOD 2: the indexed form returns the same element of the same map
    for i = 1:(Nchan*Ntimes)
        @test PermutationTests.testStatistic(
            signs,
            Y,
            i,
            stat;
            adj = adj,
            Nchan = Nchan,
            Ntimes = Ntimes,
        ) ≈ score[i] rtol = 1e-6
    end
end

@testitem "exact enumeration matches Monte-Carlo counting" tags =
    [:permutation, :validation] setup = [PermHelpers] begin
    rng = MersenneTwister(31)
    Nchan, Ntimes, n = 5, 5, 8  # 2^8 = 256 sign patterns
    data = randn(rng, Nchan, Ntimes, n)
    data[2:4, 2:4, :] .+= 1.2
    adj = path_adjacency(Nchan)

    nperm = 2048
    pfwe_e, pun_e = permutation_test(data, adj; exact = true, method = :count)
    pfwe_m, pun_m = permutation_test(data, adj; nperm = nperm, seed = 77, method = :count)

    # exact p-values are (count + 1) / 2^9, integer multiples of 1/257, and at
    # least 2/257: the identity sign pattern is one of the enumerated
    # permutations and reproduces the observed statistic exactly, so every
    # count includes it
    for p in (pun_e, pfwe_e)
        @test all(abs.(p .* 257 .- round.(p .* 257)) .< 1e-9)
        @test minimum(p) >= 2 / 257 - 1e-12
    end

    # the Monte-Carlo counts estimate the exact probabilities: within 3 standard
    # errors of the binomial plus the (count+1)/(n+1) bias gap of the two
    # estimators, element by element (fixed seed, hence deterministic)
    for i = 1:(Nchan*Ntimes)
        for (pe, pm) in [(pun_e[i], pun_m[i]), (pfwe_e[i], pfwe_m[i])]
            se = sqrt(pe * (1 - pe) / nperm)
            gap = (1 - pe) * abs(1 / 257 - 1 / (nperm + 1))
            @test abs(pm - pe) <= 3 * se + gap
        end
    end
end

@testitem "Gamma tail fit: matches counting, resolves below the floor" tags =
    [:permutation, :validation] setup = [PermHelpers] begin
    # where counting is reliable, the Gamma fit must agree with it
    rng = MersenneTwister(10)
    null = gumbel_max(rng, 20000)
    ref = sort(null, rev = true)
    for q in (0.05, 0.02, 0.01)  # thresholds whose true p-value we know from the big sample
        x = ref[Int(floor(q*length(ref)))+1]
        p = gamma_pvalue([x], null[1:1000])[1]
        se = sqrt(q * (1 - q) / 1000)  # 3 standard errors of the counting estimate
        @test abs(p - q) < 3 * se + 0.25 * q
    end

    # from 500 permutations, counting cannot go below 1/500 = 0.002
    rng = MersenneTwister(11)
    null = gumbel_max(rng, 500)
    x = maximum(null) + 2.0  # far out in the tail
    p = gamma_pvalue([x], null)[1]
    @test 0 < p <= 1 / 500
    @test isfinite(p)

    # never returns zero
    rng = MersenneTwister(12)
    null = gumbel_max(rng, 1000)
    p = gamma_pvalue([maximum(null) + 10.0], null)[1]
    @test p > 0
end

@testitem "GPD tail fit: beats counting below the floor" tags =
    [:permutation, :validation, :slow] setup = [PermHelpers] begin
    # a real permutation distribution: a one-sample t under sign-flipping
    # (mirrors TestPareto in test_tails.py)
    rng = MersenneTwister(13)
    n, n_elem, n_ref = 40, 400, 20000
    Y = randn(rng, n_elem, n)

    null = Matrix{Float64}(undef, n_ref, n_elem)
    @inbounds for p = 1:n_ref
        s = (rand(rng, n) .< 0.5) .* 2.0 .- 1.0  # a fresh ±1 sign vector
        Ys = Y .* reshape(s, 1, :)
        m = mean(Ys, dims = 2)
        sd = std(Ys, dims = 2; corrected = true)
        null[p, :] = vec(m ./ (sd / sqrt(n)))
    end

    n_acc = 1000  # what the fit is allowed to see
    ref = -sort(-null, dims = 1)
    for q in (3e-3, 1e-3, 3e-4)
        x = ref[Int(floor(q*n_ref))+1, :]  # threshold with true p-value q
        seen = @view null[1:n_acc, :]
        cnt = vec(sum(seen .>= x[1:1, :], dims = 1))
        p_cnt = cnt / n_acc

        # the fit only ever needs the tail, but the count has to come from every
        # permutation -- a truncated tail cannot say how often an element was
        # exceeded, which is the whole of the p-value
        tail = -sort(-seen, dims = 1)[1:100, :]
        p_par = pareto_pvalue(x, tail, cnt, n_acc)

        # unbiased in the middle
        @test 0.5 < median(p_par ./ q) < 2.0

        # never zero, which counting is, constantly, down here
        @test all(p_par .> 0)

        # and at least as often within a factor of two of the truth as counting
        within2(pp) = mean((pp ./ q .>= 0.5) .& (pp ./ q .<= 2.0))
        @test within2(p_par) >= within2(p_cnt) - 0.05
    end

    # elements counting already resolves are left alone
    seen = @view null[1:n_acc, :]
    x = ref[Int(floor(0.2*n_ref))+1, :]  # p ~ 0.2, so ~200 exceedances
    cnt = vec(sum(seen .>= x[1:1, :], dims = 1))
    tail = -sort(-seen, dims = 1)[1:100, :]
    p = pareto_pvalue(x, tail, cnt, n_acc; n_exc_min = 25)
    @test p == cnt / n_acc
end

@testitem "GPD: PWM recovers parameters, sf is a survival function" tags =
    [:permutation, :unit, :validation] setup = [PermHelpers] begin
    # the moment estimator must find the shape and scale it was given
    # (inverse-CDF sampling, so no extra dependency)
    function gpd_sample(rng, k_true, sigma_true, m, n)
        u = rand(rng, m, n)
        if abs(k_true) < 1e-8
            y = -log.(u) * sigma_true
        else
            y = (sigma_true / k_true) .* (1 .- (1 .- u) .^ k_true)
        end
        return sort(y, dims = 1)
    end
    for (k_true, sigma_true) in [(0.2, 1.5), (-0.3, 2.0), (0.0, 1.0)]
        rng = MersenneTwister(20)
        y = gpd_sample(rng, k_true, sigma_true, 4000, 200)
        k, sigma = gpd_fit_pwm(y)
        @test abs(median(k) - k_true) < 0.05
        @test abs(median(sigma) - sigma_true) < 0.1
    end

    # the survival function survives
    z = range(0, 5, length = 50)
    for k in (-0.3, 0.0, 0.3)
        s = gpd_sf(z, fill(k, 50), ones(50))
        @test s[1] ≈ 1.0
        @test all((s[2:end] .- s[1:(end-1)]) .<= 1e-12)
        @test all(s .>= 0) && all(s .<= 1)
    end

    # k > 0 bounds the distribution above -- beyond the end point it says zero
    k, sigma = 0.5, 1.0
    @test gpd_sf([sigma / k + 1.0], [k], [sigma])[1] == 0.0
end

@testitem "null calibration: uniform uncorrected p, FWE control" tags =
    [:permutation, :validation, :slow] setup = [PermHelpers] begin
    Nchan, Ntimes, n = 12, 8, 8
    R = 20
    nperm = 512
    adj = path_adjacency(Nchan)

    # (a function: top-level for loops have soft scope, and `nrej +=` in one
    # would shadow the outer variable)
    function run_null_calibration()
        pun_all = Float64[]
        nrej = 0
        for r = 1:R
            data = randn(MersenneTwister(1000 + r), Nchan, Ntimes, n)  # pure null
            pfwe, pun = permutation_test(
                data,
                adj;
                nperm = nperm,
                seed = 10000 + r,
                method = :count,
            )
            append!(pun_all, pun[:])
            nrej += (minimum(pfwe) < 0.05)
        end
        return pun_all, nrej
    end
    pun_all, nrej = run_null_calibration()

    # FWE control at 0.05: the expected number of false rejections is R*0.05 = 1
    @test nrej <= 8
    # the uncorrected p-values are uniform under the null
    @test abs(mean(pun_all) - 0.5) < 0.05
    @test abs(std(pun_all) - 1 / sqrt(12)) < 0.05
end

@testitem "end to end: the :tail method respects the floor and the fallback" tags =
    [:permutation, :validation] setup = [PermHelpers] begin
    rng = MersenneTwister(41)
    Nchan, Ntimes, n = 8, 6, 6
    data = randn(rng, Nchan, Ntimes, n)
    data[3:5, 2:4, :] .+= 1.5  # planted effect
    adj = path_adjacency(Nchan)

    nperm = 200
    pfwe_c, pun_c = permutation_test(data, adj; nperm = nperm, seed = 99, method = :count)
    pfwe_t, pun_t = permutation_test(data, adj; nperm = nperm, seed = 99, method = :tail)

    # shape, finiteness, range: the Gamma fit clips to [1/nperm, 1], the GPD fit
    # never returns zero or one
    @test size(pfwe_t) == size(pun_t) == (Nchan, Ntimes)
    @test all(isfinite, pfwe_t) && all(isfinite, pun_t)
    @test all(pfwe_t .>= 1 / nperm - 1e-12) && all(pfwe_t .<= 1)
    @test all(pun_t .> 0) && all(pun_t .<= 1)

    # elements counting already resolves (at least 25 exceedances in 200) are left
    # alone by the GPD fit, which returns the plain count cnt/nperm for them. The
    # :count map uses the (cnt+1)/(nperm+1) convention, so invert it to recover
    # cnt and check against that instead
    cnt = round.((nperm + 1) .* vec(pun_c) .- 1)
    well_counted = cnt .>= 25
    @test any(well_counted) && !all(well_counted)
    @test all(pun_t[well_counted] .== cnt[well_counted] ./ nperm)
end

@testitem "cross-language parity with the Python reference" tags =
    [:permutation, :crosslang, :validation] setup = [PermHelpers] begin
    # Fixtures from test/parity/make_parity.py, which runs the reference toolbox
    # (PYTHONPATH=.../tfce/python/src) and dumps the inputs and the reference
    # outputs as float64 little-endian .bin files.
    dir = (@__DIR__) * "/parity"
    for bin in ("onesamp.bin", "tails.bin")
        isfile(dir * "/" * bin) || error(
            "missing parity fixture $(bin); regenerate it with " *
            "PYTHONPATH=<tfce source>/src python test/parity/make_parity.py",
        )
    end

    # 1. the one-sample t of a fixed sign-flip: the reference's PermutedGLM.fit_signs
    let f = open(dir * "/onesamp.bin", "r")
        Yt = reshape(readbin(f, 120 * 12), (12, 120))  # Yt[s, e] = Y[e, s]
        signs = readbin(f, 12)
        t_ref = readbin(f, 120)
        close(f)
        t_pkg = [onesamp_t(Yt[:, e] .* signs) for e = 1:120]
        @test maximum(abs, t_pkg .- t_ref) < 1e-12
    end

    # 2. the tail fits: identical inputs, p-values to floating-point precision
    let f = open(dir * "/tails.bin", "r")
        stat = readbin(f, 120)
        null_max = readbin(f, 512)
        tail = reshape(readbin(f, 120 * 100), (100, 120))
        cnt = readbin(f, 120)
        n_perm = Int(readbin(f, 1)[1])
        p_g_ref = readbin(f, 120)
        p_p_ref = readbin(f, 120)
        close(f)

        @test maximum(abs, gamma_pvalue(stat, null_max) .- p_g_ref) < 1e-9
        @test maximum(abs, pareto_pvalue(stat, tail, cnt, n_perm) .- p_p_ref) < 1e-9
    end
end
