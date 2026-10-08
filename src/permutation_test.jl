"""
    permutation_test(data, adjacency;
                     E = 0.5, H = 2.0, two_sided = true,
                     nperm = 1000, seed = 1234,
                     method = :tail, exact = false,
                     tail_size = 100, n_exc_min = 25, n_shape = 20000, ad_max = 5.0)
        -> (p_fwe, p_uncorrected)

One-sample sign-flip permutation test of a TFCE score map.

`data` is `channels × times × subjects`; for each (channel, time) element the
one-sample *t*-statistic across subjects is the height, and TFCE over the
channel neighbourhood (see [`tfce`](@ref)) is the spatial statistic, so the
test statistic of an element is the TFCE score at that (channel, time). The
permutation mechanism is the sign-flip scheme of PermutationTests.jl's
`OneSampStatistic` (`StudentT_1S`): each permutation multiplies every element's
subject values by a fresh random sign vector (drawn with `MersenneTwister(seed)`,
so the result is reproducible for a given `seed`).

Two p-value maps are returned, each with the shape of the TFCE score map
(channels × times), both two-sided in magnitude (`|score|`):

* `p_fwe` — FWE-corrected, from the null of the *maximum* statistic over all
  elements.
* `p_uncorrected` — uncorrected, from each element's own null.

`method` selects the p-value machinery:

* `:tail` (default) — Winkler tail fits: a Gamma moment-fit of the
  maximum-statistic null for `p_fwe` ([`gamma_pvalue`](@ref)) and a Generalised
  Pareto fit (pooled shape, Anderson-Darling gate) of each element's tail for
  `p_uncorrected` ([`pareto_pvalue`](@ref)). These resolve p-values below the
  `1/nperm` counting floor; each falls back to counting when a fit degenerates.
* `:count` — plain counting with the `(count + 1) / (nperm + 1)` convention:
  the observed value counts as one of the `nperm + 1` exchangeable values, so
  p-values lie in `[1/(nperm + 1), nperm/(nperm + 1)]` — never `0`, resolving
  down to `1/(nperm + 1)`.

`exact = true` enumerates all `2^n` sign patterns instead of drawing `nperm`
of them (cheap only for small `n`); the counting p-values are then exact.
`tail_size` is the number of largest per-element null values kept for the
uncorrected fit (the reference finds ~100 plenty); `n_exc_min`, `n_shape` and
`ad_max` pass through to [`pareto_pvalue`](@ref).
"""
function permutation_test(
    data,
    adjacency;
    E = 0.5,
    H = 2.0,
    two_sided = true,
    nperm = 1000,
    seed = 1234,
    method = :tail,
    exact = false,
    tail_size = 100,
    n_exc_min = 25,
    n_shape = 20000,
    ad_max = 5.0,
)
    method in (:tail, :count) ||
        throw(ArgumentError("method must be :tail or :count, got $(repr(method))"))

    ndims(data) == 3 || throw(
        ArgumentError("data must be channels × times × subjects, got $(ndims(data))-D"),
    )
    Nchan, Ntimes, nsubj = size(data)
    nsubj >= 2 ||
        throw(ArgumentError("a one-sample test needs at least 2 subjects, got $nsubj"))
    (size(adjacency, 1) == Nchan && size(adjacency, 2) == Nchan) ||
        throw(ArgumentError("adjacency must be $(Nchan) × $(Nchan) for $Nchan channels"))
    exact || nperm >= 2 || throw(ArgumentError("nperm must be >= 2, got $nperm"))
    exact &&
        nsubj > 20 &&
        throw(
            ArgumentError(
                "exact enumeration needs 2^nsubj patterns to stay manageable; nsubj = $nsubj, use exact = false",
            ),
        )

    # elements in column-major (channel fastest) order, matching the layout of a
    # channels × times score map: element i is channel (i-1) % Nchan + 1,
    # time (i-1) ÷ Nchan + 1
    M = Nchan * Ntimes
    Yorig = [Vector{Float64}(data[(i-1)%Nchan+1, (i-1)÷Nchan+1, :]) for i = 1:M]

    stat = TFCEOneSampT()
    x = PermutationTests.membership(PermutationTests.StudentT_1S(), nsubj)

    # observed statistic: the unpermuted score map, in magnitude
    score0 = tfce_score_map(
        Yorig;
        adj = adjacency,
        Nchan = Nchan,
        Ntimes = Ntimes,
        E = E,
        H = H,
        two_sided = two_sided,
    )
    obs = abs.(Float64.(reshape(score0, :)))

    n_used = exact ? 2^nsubj : nperm
    K = min(tail_size, n_used)
    acc = NullAccumulator(M, n_used, K)
    s = Vector{Float64}(undef, M)

    if exact
        _run_null_exact!(
            Yorig,
            x,
            stat,
            acc,
            s,
            obs,
            adjacency,
            Nchan,
            Ntimes,
            E,
            H,
            two_sided;
            n = nsubj,
        )
    else
        # `_run_null!` sign-flips its argument in place (cumulatively). `Yorig`
        # is a local buffer — the observed statistic was already taken from it
        # and nothing reads it after the null run — so it serves as the
        # working copy and no second copy is made.
        rng = Random.MersenneTwister(seed)
        _run_null!(
            Yorig,
            x,
            stat,
            acc,
            s,
            obs,
            adjacency,
            Nchan,
            Ntimes,
            E,
            H,
            two_sided;
            nperm = nperm,
            rng = rng,
        )
    end

    if method == :count
        # (count + 1) / (n_used + 1): the observed value counts as one of the
        # n_used + 1 exchangeable values, so p-values never reach 0
        null_max = acc.null_max
        p_fwe = _count_sf(obs, null_max; plus_one = true)
        p_unc = (vec(acc.cnt) .+ 1) ./ (n_used + 1)
    else
        # tail: (K, M) — the largest per-element null values, only the tail
        # fit needs them
        tail = Matrix{Float64}(undef, K, M)
        @inbounds for i = 1:M
            copyto!(view(tail, :, i), acc.tails[i])
        end
        p_fwe = gamma_pvalue(obs, acc.null_max)
        p_unc = pareto_pvalue(
            obs,
            tail,
            vec(acc.cnt),
            n_used;
            n_exc_min = n_exc_min,
            n_shape = n_shape,
            ad_max = ad_max,
        )
    end

    return reshape(p_fwe, Nchan, Ntimes), reshape(p_unc, Nchan, Ntimes)
end
