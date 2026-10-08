# Port of the reference toolbox's `tfce/tails.py` (Winkler et al., Faster
# permutation inference in brain imaging, NeuroImage 141:502-516, 2016):
# a Gamma moment-fit for the FWE p-values of the maximum statistic, and a
# Generalised Pareto fit (probability-weighted moments, pooled shape,
# Anderson-Darling acceptance gate) for the uncorrected p-values of each
# element's own permutation tail. Both fall back to counting when a fit has
# nothing to hold on to, mirroring the reference.

"""
    gamma_pvalue(stat, null_max; floor = 1/length(null_max))

FWE-corrected p-values from a Gamma fit to the null of the *maximum* statistic.

`stat` holds the observed statistic, one value per element; `null_max` holds
the maximum of the statistic over all elements, one value per permutation.
A Gamma with shape `k`, scale `theta` and location `mu` is fitted by matching
the first three moments in closed form

    mean = mu + k*theta,   var = k*theta^2,   skew = 2/sqrt(k),

so this is cheap and stable. If the sample skewness comes out non-positive (or
the variance is not positive) there is nothing for a Gamma to hold on to and
the empirical count `P(null_max >= stat)` is returned instead. The fitted
p-values are clipped to `[floor, 1]`; the default floor of `1/n_perm` is what
counting could have resolved, so the fit is never asked to extrapolate further
than the permutations can support.
"""
function gamma_pvalue(stat, null_max; floor = nothing)
    stat = vec(Float64.(stat))
    null_max = vec(Float64.(null_max))
    n_perm = length(null_max)
    isnothing(floor) && (floor = 1.0 / n_perm)

    m = mean(null_max)
    v = var(null_max; corrected = true)
    skew = _sample_skew(null_max)

    if !isfinite(skew) || skew <= 0 || v <= 0
        # nothing for a Gamma to fit; fall back to counting
        return _count_sf(stat, null_max)
    end

    k = 4.0 / skew^2
    theta = sqrt(v / k)
    mu = m - k * theta

    p = _gamma_sf.(stat, k, mu, theta)
    return clamp.(p, floor, 1.0)
end

# Sample skewness, the `bias = false` convention of scipy.stats.skew: the
# adjusted Fisher-Pearson coefficient G1 = sqrt(n(n-1))/(n-2) * m3/m2^(3/2),
# with m2, m3 the (biased) central moments.
function _sample_skew(x::AbstractVector{<:Real})
    n = length(x)
    n >= 3 || return NaN
    m = mean(x)
    d = x .- m
    m2 = sum(d .^ 2) / n
    m3 = sum(d .^ 3) / n
    return (m3 / m2^1.5) * sqrt(n * (n - 1)) / (n - 2)
end

# Upper tail of a Gamma(shape = a, scale = theta) shifted by `mu`, at `x`:
# `scipy.stats.gamma.sf(x, a = a, loc = mu, scale = theta)`. This is the
# regularized upper incomplete gamma Q(a, (x - mu)/theta); for (x - mu) <= 0
# the observation is left of the location and the survival is 1.
function _gamma_sf(x::Real, a::Real, mu::Real, theta::Real)
    z = (x - mu) / theta
    z <= 0 && return 1.0
    # gamma_inc(a, z) -> (P(a, z), Q(a, z)); the survival is the upper, Q.
    return gamma_inc(a, z)[2]
end

# P(null >= stat), by counting (the reference's `_count_sf`). With
# `plus_one = true` the observed value counts as one of the `n + 1`
# exchangeable values, giving `(n_ge + 1) / (n + 1)`, which never reaches 0.
function _count_sf(stat, null; plus_one = false)
    nulls = sort(vec(Float64.(null)))
    n = length(nulls)
    n_ge = n .- (searchsortedfirst.(Ref(nulls), stat) .- 1)
    return plus_one ? (n_ge .+ 1) ./ (n + 1) : n_ge ./ n
end

"""
    gpd_fit_pwm(y)

Generalised Pareto parameters of `y` (shape (m, n): exceedances above a
threshold, ascending down each column, one column per element) by
probability-weighted moments, as in Hosking & Wallis (1987):

    F(y) = 1 - (1 - k*y/sigma)^(1/k).

Closed form, hence every element is fitted at once. Returns `(k, sigma)`,
each of length n.
"""
function gpd_fit_pwm(y::AbstractMatrix{<:Real})
    m = size(y, 1)
    j = collect(1:m)
    w = reshape((m .- j) ./ (m * (m - 1)), m, 1)
    a0 = vec(Float64.(mean(y, dims = 1)))
    a1 = vec(Float64.(sum(y .* w, dims = 1)))
    d = a0 .- 2a1
    k = @. a0 / d - 2
    sigma = @. 2a0 * a1 / d
    return k, sigma
end

"""
    gpd_sf(z, k, sigma)

Survival function of the Generalised Pareto, elementwise. `k -> 0` is
evaluated as its exponential limit; beyond the finite end point of a
`k > 0` fit the survival is exactly zero.
"""
function gpd_sf(z, k, sigma)
    z = Float64.(z)
    k = Float64.(k)
    sigma = Float64.(sigma)
    # Julia aligns broadcast dimensions from the FIRST axis (unlike NumPy), so a
    # per-element parameter vector must be a row to apply down the columns of z.
    if ndims(z) == 2 && ndims(k) == 1
        k = reshape(k, 1, :)
        sigma = ndims(sigma) == 1 ? reshape(sigma, 1, :) : sigma
    end
    return _gpd_sf_elem.(z, k, sigma)
end

function _gpd_sf_elem(zv::Real, kv::Real, sv::Real)
    if abs(kv) < 1e-8                        # the k -> 0 limit is exponential
        return exp(-zv / sv)
    end
    q = 1 - kv * zv / sv
    q < 0 && (q = 0.0)                       # beyond the end point of a k > 0 fit
    return q^(1.0 / kv)
end

# Anderson-Darling statistic of a fitted GPD against the tail it was fitted to.
# Used only to rank candidate tail sizes against each other, so what matters
# is that it grows when the fit describes the observed tail less well.
function _gpd_anderson(y, k, sigma)
    m = size(y, 1)
    j = reshape(collect(1:m), m, 1)
    F = 1 .- gpd_sf(y, k, sigma)
    F = clamp.(F, 1e-12, 1 - 1e-12)
    return -m .-
           vec(sum(((2 .* j) .- 1) .* (log.(F) .+ log.(1 .- F[end:-1:1, :])), dims = 1)) / m
end

"""
    pareto_pvalue(stat, tail, cnt, n_perm; n_exc_min = 25, n_shape = 20000, ad_max = 5.0)

Uncorrected p-values, resolved below the `1/n_perm` floor of counting, by
fitting a Generalised Pareto to the tail of each element's own permutation
distribution.

`stat` is the observed statistic, one value per element; it must already be
an *upper* tail (flip the sign of the elements whose effect is negative
before calling, so that both sides ask the same question). `tail` has shape
(K, n_elements): the largest `K` permuted values of each element, same sign
convention. `cnt` is how often the permutations reached or exceeded `stat`,
counted over **all** `n_perm` permutations -- a truncated tail cannot answer
that question, which is why it is passed in rather than derived.

A GPD's *shape* is pooled across elements: they all carry the same statistic
under the same design, so they differ in scale, not in shape, and fitting the
shape per element would spend a hundred tail values on a number they all
share. Each candidate tail size is accepted only if its Anderson-Darling
statistic against the observed tail stays below `ad_max` (deliberately loose:
it catches a fit that has gone wrong, not a hypothesis about the tail).
Elements whose statistic was exceeded at least `n_exc_min` times are left to
plain counting: the count is then already precise enough.
"""
function pareto_pvalue(
    stat,
    tail,
    cnt,
    n_perm;
    n_exc_min = 25,
    n_shape = 20000,
    ad_max = 5.0,
)
    stat = vec(Float64.(stat))
    tail = Matrix{Float64}(tail)
    cnt = vec(Float64.(cnt))
    n_perm = Int(n_perm)

    K, n_elem = size(tail)
    length(stat) == n_elem ||
        throw(ArgumentError("stat has $(length(stat)) elements but tail describes $n_elem"))
    length(cnt) == n_elem ||
        throw(ArgumentError("cnt has $(length(cnt)) elements but tail describes $n_elem"))

    p = cnt ./ n_perm

    K < 30 && return p

    # only the elements counting cannot resolve are worth fitting
    need = (cnt .< n_exc_min) .& isfinite.(stat)
    any(need) || return p

    # descending tails: the fitted elements, and a sample of ALL of them for the
    # shape. The shape is pooled over the whole image on purpose -- the elements
    # being fitted were selected for carrying a large statistic, and selecting on
    # that would bias the tails it is estimated from.
    G = -sort(-tail, dims = 1)
    Gn = G[:, need]
    ob = stat[need]

    step = max(1, n_elem ÷ n_shape)
    Gs = G[:, 1:step:n_elem]

    best_ad = fill(Float64(Inf), length(ob))
    best_p = copy(p[need])

    cands = sort([
        m for m in Set{Int}(round(Int, K * f) for f = 0.9:-0.1:0.29) if 20 <= m <= K - 1
    ])

    for m in cands
        # pooled shape, from the sampled elements
        us = (Gs[m, :] + Gs[m+1, :]) / 2
        ys = Gs[m:-1:1, :] .- reshape(us, 1, :) # ascending exceedances
        ks, _ = gpd_fit_pwm(ys)
        ks = ks[isfinite.(ks)]
        isempty(ks) && continue
        k_pool = Float64(median(ks))

        u = (Gn[m, :] + Gn[m+1, :]) / 2
        y = Gn[m:-1:1, :] .- reshape(u, 1, :)

        # the pooled shape, and the exponential limit as a safety net: a GPD with
        # k > 0 has a finite upper end point at u + sigma/k, and where the observed
        # statistic lies beyond it the fit returns exactly zero. The observation is
        # itself proof that the end point is wrong, so such a fit is rejected
        # rather than believed. The exponential has infinite support and always
        # answers.
        if isfinite(k_pool) && k_pool > -1
            variants = [fill(k_pool, length(ob)), zeros(length(ob))]
        else
            variants = [zeros(length(ob))]
        end

        for kk in variants
            # given the shape, the scale follows from the mean of the exceedances,
            # because the GPD has mean sigma/(1 + k)
            sigma = vec(mean(y, dims = 1)) .* (1 .+ kk)
            pp = (m / n_perm) .* gpd_sf(ob .- u, kk, sigma)
            ad = _gpd_anderson(y, kk, sigma)
            ok =
                (sigma .> 0) .& (ob .>= u) .& isfinite.(pp) .& (pp .> 0) .& (pp .<= 1) .&
                isfinite.(ad)
            upd = ok .& (ad .< best_ad)
            best_ad = ifelse.(upd, ad, best_ad)
            best_p = ifelse.(upd, pp, best_p)
        end
    end

    trust = isfinite.(best_ad) .& (best_ad .< ad_max)
    out = copy(p)
    out[need] = ifelse.(trust, best_p, p[need])
    return out
end
