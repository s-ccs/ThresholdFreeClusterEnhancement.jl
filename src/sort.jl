# --------------------------------------------------------------------------------------
# Sorting. `sortalg = :quick` (the default) just uses `Base`'s quicksort; everything in
# this file except that one line is the experimental `sortalg = :bucket` path.
#
# Every value reaching the sort is strictly positive, and for positive IEEE-754 floats
# the bit pattern read as a `UInt64` increases monotonically with the value. The pattern
# is `exponent << 52 | mantissa`, so *all keys sharing an exponent are contiguous in
# sorted order*: bucketing on the exponent is an exact partition, not a quantisation,
# and only the handful of keys inside a bucket still need comparing. Exponent spans
# within a single map are tiny -- measured on the benchmark data, median 5-8 and max 22
# -- so the histogram is cheap to clear and buckets hold a few elements each.
#
# Below `_BUCKET_MIN` a plain insertion sort wins: the counting sort's fixed costs
# (clearing the histogram, the extra counting pass) do not pay for themselves when there
# are only a few keys. A wide exponent span (only reachable for data spanning many
# binades) falls back to `QuickSort`, which is the one Base algorithm that does not
# allocate.
#
# Equal values are not a problem for the sweep: it consumes a whole tie group as one
# level, and both the resulting partition and every element's score are independent of
# the order the group is merged in.
# --------------------------------------------------------------------------------------
const _BUCKET_MIN = 40
const _MAX_EXPSPAN = 24

# Sort `w.ord[1:n]` ascending by value; `:quick` and the bucket fallback both use
# `Base.Sort.QuickSort`, which does not allocate.
@inline function _sort_positive!(w::Workspace, n::Int)
    if w.sortalg === :bucket
        if n <= _BUCKET_MIN
            _insertion_sort!(w.ord, 1, n)
        elseif !_bucket_sort!(w.ord, w.ord_tmp, w.cnt, n, _MAX_EXPSPAN)
            sort!(w.ord, 1, n, Base.Sort.QuickSort, Base.Order.Forward)
        end
    else
        sort!(w.ord, 1, n, Base.Sort.QuickSort, Base.Order.Forward)
    end
    return n
end

# Exponent field of a positive double (bits 52..62).
@inline _expof(x::Float64) = Int((reinterpret(UInt64, x) >> 52) & 0x7ff)

"""
    _bucket_sort!(ord, ord_tmp, cnt, n, maxspan) -> Bool

Counting sort of `ord[1:n]` by key exponent into `ord_tmp`, then insertion sort inside
each bucket, then copy back. Returns `false` without touching `ord` if the exponent span
exceeds `maxspan`, leaving the caller to fall back to a comparison sort. `cnt` must hold
at least `maxspan + 1` entries.
"""
function _bucket_sort!(ord, ord_tmp, cnt, n::Int, maxspan::Int)
    @inbounds begin
        emin = _expof(ord[1][1])
        emax = emin
        for i = 1:n
            e = _expof(ord[i][1])
            e < emin && (emin = e)
            e > emax && (emax = e)
        end
        span = emax - emin
        span > maxspan && return false
        nb = span + 1
        for k = 1:nb
            cnt[k] = 0
        end
        for i = 1:n                                       # histogram
            cnt[_expof(ord[i][1])-emin+1] += 1
        end
        s = 1
        for k = 1:nb                                      # exclusive prefix sum:
            c = cnt[k]                                    # cnt[k] becomes bucket k's
            cnt[k] = s                                    # first slot
            s += c
        end
        for i = 1:n                                       # scatter, bucketed by exponent
            b = _expof(ord[i][1]) - emin + 1
            j = cnt[b]
            ord_tmp[j] = ord[i]
            cnt[b] = j + 1
        end
        lo = 1                                            # after the scatter cnt[k] is one
        for k = 1:nb                                      # past the end of bucket k
            hi = cnt[k] - 1
            _insertion_sort!(ord_tmp, lo, hi)
            lo = hi + 1
        end
    end
    copyto!(ord, 1, ord_tmp, 1, n)
    return true
end

# Insertion sort of `ord[lo:hi]` by key. `>` compares keys only, which leaves equal
# values in the order they were gathered in.
@inline function _insertion_sort!(ord, lo::Int, hi::Int)
    @inbounds for i = (lo+1):hi
        x = ord[i]
        j = i - 1
        while j >= lo && ord[j][1] > x[1]
            ord[j+1] = ord[j]
            j -= 1
        end
        ord[j+1] = x
    end
    return ord
end
