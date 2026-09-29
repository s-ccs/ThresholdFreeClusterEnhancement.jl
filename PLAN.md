# TFCE implementation plan

Reference implementation: `~/tfce/tfce` (C core `tfce_maxtree.h`, Python wrapper `core.py`).
This plan: exact TFCE via max-tree + union-find in Julia, no toolbox dependencies (no SPM, no scipy).

---

## Scope

- Target data: EEG/MEG-style `channels × times × subjects`, plus a channel-neighbourhood matrix.
- Implement **only the explicit-adjacency graph path** (the C "mesh" path): `N = n_channels`, one independent map per (time, subject).
- Deferred: volume grid path (6/18/26-connectivity), threads, batching, radix sort, permutation GLM, p-value machinery.
- API:

```julia
tfce(data::AbstractArray{<:Real,3}, adjacency::AbstractMatrix;
     E = 0.5, H = 2.0, two_sided = true) -> Array{Float32,3}   # channels × times × subjects
tfce(data::AbstractMatrix, adjacency; …)                        # channels × times → same size
```

- `adjacency`: channels × channels (0/1 or `Bool`, symmetric) → converted once to CSR-like adjacency lists.
- Each time point is a separate 1-D map over channels; subjects processed independently.
- No dependencies beyond stdlib.

**Future-proofing (time-frequency):** the core stays a *graph* TFCE: flat vector of `N` maps + adjacency lists. The 4-D case (channels × times × freqs × subjects) becomes a thin wrapper: loop freqs, or confluate channel & frequency neighbourhoods by concatenating adjacencies (block-diagonal across freqs if clusters must not cross frequency boundaries; cross-linked rows if they may). Design constraint: don't bake "channels" into the core.

---

## Algorithm (extracted from the reference)

**TFCE definition.** For each element v: `TFCE(v) = ∫₀^{t_v} e_v(h)^E · h^H dh`, where `e_v(h)` = size of the connected component containing v when thresholding at height h. Classic implementations step h over a grid (Riemann sum, O(dh) error, tunable). This reference computes it **exactly**.

**Two observations make it exact:**

1. **The integrand is piecewise constant.** As h decreases, the component containing v only changes at merge heights, so v's component history is a finite list of pieces `(size s_k, interval [death_k, birth_k))`, each integrating in closed form: `s_k^E · (birth_k^(H+1) − death_k^(H+1))/(H+1)`.
2. **The pieces are nested → a tree.** The component containing v at h₁ ⊇ the component at h₂ > h₁. Every element's chain of components is a path in one shared tree, the **max-tree**:
   - **Node** = one component, alive over `[death, birth)`: born when an element was added to it or it formed by a merge; dies when it merges into a bigger one.
   - **Parent** = the component it merges into (strictly bigger, strictly lower birth).
   - **Leaf** = the first component containing an element (born at the element's own height).
   - **Root** = the final component (born at the lowest positive height, death = 0).
   - A child's death = parent's birth, so the pieces on a root→leaf path **tile (0, t_v] exactly** and the sum telescopes to the true integral: `TFCE(v) = Σ over v's path of node contributions`.

**The sweep** (one pass over the positive values):

1. Sort positive elements descending; process levels of ties (equal values together).
2. **Activate** tie elements as singletons.
3. **Union** each with its already-active neighbours (path-halving find + union-by-size). Descending order ⇒ unioning captures exactly the merges at this height; the union-find forest *is* the component partition of the active set, maintained incrementally.
4. **Emit one tree node per changed component** (birth = h); mark all alive nodes under its root dead (death = h, parent = new node). Each root keeps a list of its alive nodes, spliced O(1) on union.
5. Assign each element its leaf node. Node indices are creation-ordered and parent index > child index ⇒ one backward loop accumulates `cum` root→leaves.

Two-sided = same procedure on `−d`, negated; the passes touch disjoint elements. Only `d > 0` enters the tree; all-zero map → all-zero output. Per-element values are `Float32`-worthy; the integral accumulation must be `Float64` (`birth^(H+1)` spans a wide range). Complexity: O(N log N) sort + O(N α(N)) sweep, independent of any step-size parameter.

**Worked example** (line adjacency 1–2–3–4, values `d = [0, 3, 1, 2]`, `E = 1, H = 0`; a node contributes `s·(birth − death)`):

| level h | activate | unions | node emitted |
|---|---|---|---|
| 3 | {2} | none | A: s=1, [3, 1) |
| 2 | {4} | none | B: s=1, [2, 1) |
| 1 | {3} | 3–2, 3–4 | C: s=3, [1, 0); **A, B die here** |

Tree: root C(s=3, [1,0)), children A(s=1,[3,1)), B(s=1,[2,1)). Contributions: A=2, B=1, C=3. Cumulated: cum[C]=3, cum[A]=5, cum[B]=4. Elements map to their leaf: v2→A, v3→C, v4→B → **TFCE = [0, 5, 3, 4]**. Cross-check (element 2): ∫₀³ e(h)dh = 2·1 + 1·3 = 5 ✓.

---

## Julia pseudocode

```julia
# ---- one signed sweep over the positive part of d, accumulating into out ----
function tfce_pass!(out, d, adj, E, H, sign)
    N = length(d)
    ord = [i for i in 1:N if d[i] > 0]           # positive elements only
    isempty(ord) && return
    sort!(ord; by = i -> d[i], rev = true)       # descending by value

    active = falses(N)                           # element is above current height
    parent = collect(1:N)                        # union-find forest (lazy)
    sz     = ones(Int, N)                        # union by size
    root_stamp = fill(0, N)                      # level each root last got a node
    elem_node  = zeros(Int, N)
    level  = 0

    # max-tree nodes, appended as created
    nd_parent = Int[];   nd_size  = Int[]
    nd_birth  = Float64[]; nd_death = Float64[]
    alive     = [Int[] for _ in 1:N]             # per root: list of its alive nodes

    i = 1
    while i <= length(ord)                       # ties (equal values) = one level
        h = d[ord[i]];  j = i
        while j <= length(ord) && d[ord[j]] == h; j += 1; end
        tie = ord[i:j-1];  i = j
        level += 1

        for u in tie                             # 1. activate as singletons
            active[u] = true
            parent[u] = u;  sz[u] = 1
        end
        for u in tie, v in adj[u]                # 2. merge with active neighbours
            if active[v]
                ra, rb = findroot!(parent, u), findroot!(parent, v)
                if ra != rb
                    if sz[ra] < sz[rb]; ra, rb = rb, ra; end
                    parent[rb] = ra
                    sz[ra] += sz[rb]
                    append!(alive[ra], alive[rb]);  empty!(alive[rb])
                end
            end
        end
        for u in tie                             # 3. one node per changed component
            r = findroot!(parent, u)
            if root_stamp[r] != level
                push!(nd_parent, 0);  push!(nd_size, sz[r])
                push!(nd_birth, h);   push!(nd_death, 0.0)
                n = length(nd_size)
                for o in alive[r]                # old components of r die at h
                    nd_parent[o] = n
                    nd_death[o]  = h
                end
                alive[r] = [n]
                root_stamp[r] = level
            end
            elem_node[u] = alive[findroot!(parent, u)][1]   # u's leaf node
        end
    end

    # 4. root -> leaf accumulation (parent index > child index; Float64!)
    Hp1 = H + 1
    cum = zeros(length(nd_size))
    for n in length(nd_size):-1:1
        c = nd_size[n]^E * (nd_birth[n]^Hp1 - nd_death[n]^Hp1) / Hp1
        cum[n] = c + (nd_parent[n] > 0 ? cum[nd_parent[n]] : 0.0)
    end
    for u in ord
        out[u] += sign * cum[elem_node[u]]
    end
end

function tfce(data, adjacency; E = 0.5, H = 2.0, two_sided = true)
    adj = [filter(!=(i), unique(v)) for (i, v) in enumerate(adjacency_rows(adjacency))]
    #        ^ CSR-like adjacency lists, self-loops dropped, built once
    maps = reshape(data, size(data,1), :)        # (channels, times·subjects)
    out  = zeros(Float32, size(maps))
    for b in axes(maps, 2)                       # each (time, subject) is independent
        d = Float64.(view(maps, :, b))
        tfce_pass!(view(out, :, b), d, adj, E, H, +1)
        two_sided && tfce_pass!(view(out, :, b), -d, adj, E, H, -1)
    end
    return reshape(out, size(data))
end
```

Notes: `findroot!(parent, x)` = union-find find with path halving (`while parent[x] != x; parent[x] = parent[parent[x]]; x = parent[x]; end`). The vector-of-vectors `alive` is the readable stand-in for the C's linked lists — swap for flat arrays when we optimize. Ties via float `==` matches the reference; the two passes touch disjoint elements, so accumulating into the same `out` is safe.

---

## Tests first (`test/test-tfce.jl`, `@testitem`s, tags `[:unit, :validation]`)

No dtype tests, no shape-validation tests, no error-message tests — functionality only.

**A. Hand-computable tiny cases (exact numbers — pin down the tree/union-find machinery):**

1. 1-D line adjacency 1–2–3–4, values `[0, 3, 1, 2]`, `E=1, H=0` → exactly `[0, 5, 3, 4]`.
2. Two **connected** channels `[3, 1]`, `E=1, H=0` → exactly `[4, 2]`.
3. Two **disconnected** channels `[3, 1]`, `E=1, H=0` → exactly `[3, 1]` (tests 2+3 pin the merging behaviour).

**B. Analytic identities:**
4. `E=0` → TFCE is extent-free: `TFCE(v) = t_v^(H+1)/(H+1)` for *any* adjacency and any data (all node contributions telescope). Cheap and exact.
5. All-zero map → all-zero output.

**C. Semantics:**
6. Single spike in one channel → only that channel (one-sided) nonzero; with a connected neighbour active, the neighbour's TFCE > 0, a distant channel stays 0.
7. Two-sided: negative spike → negative TFCE; `two_sided=false` → 0; all-positive map → identical with and without `two_sided` (exact `==`).
8. Map with one positive and one negative cluster → one-sided result equals result on `max.(d, 0)` (no cross-sign leakage).

**D. The anchor test (validates the whole integral):**
9. **Convergence onto an independent reference**: naive dh-stepping TFCE (threshold grid + connected components via BFS on the *same* adjacency — deliberately naive, deliberately not ours) must converge onto the max-tree result at first order: slope of log(error) vs log(n_steps) ≈ −1 (±0.25) for n_steps ∈ {50, 100, 200, 400}, a couple of (E, H) combos, two-sided.

**E. Structural invariance:**
10. Duplicated time columns → identical results per column; 3-D call == mapping a 2-D call over subjects (subject/time independence).
11. Richer adjacency on the same data → `maximum(abs, TFCE)` grows (sanity that adjacency is actually used).

---

## Steps

1. Write the tests (they fail — function doesn't exist).
2. Implement `tfce` + `tfce_pass!` per the pseudocode in `src/ThresholdFreeClusterEnhancement.jl` until green.
3. Run tests via MCP Julia (`--project=test` workspace), `JuliaFormatter` before finishing.
