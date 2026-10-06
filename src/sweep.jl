# Sentinel for "no such node" in the per-root alive list (`head`/`tail`/`nd_next`).
const NONODE = -1

# One map, both signs. `out` is written (it is freshly zeroed by the caller).
function _tfce_map!(out, col, adj::CSRAdjacency, w::Workspace, H::Float64, two_sided::Bool)
    Hp1 = H + 1
    _tfce_pass!(out, col, adj, w, Hp1, 1.0)
    two_sided && _tfce_pass!(out, col, adj, w, Hp1, -1.0)
    return out
end

# --------------------------------------------------------------------------------------
# One signed pass over the positive part of `col`, accumulating TFCE scores into `out`
# with weight `sgn` (which is exactly ±1, so scaling the map in and the scores out is
# exact).
# --------------------------------------------------------------------------------------
function _tfce_pass!(out, col, adj::CSRAdjacency, w::Workspace, Hp1::Float64, sgn::Float64)
    N = w.N
    ord = w.ord
    n = 0                                                     # gather the positive part
    # Branchless: `x > 0` is a coin flip on noise-like data, so a conditional store here
    # mispredicts about half the time. Writing the record unconditionally and folding the
    # predicate into the counter is ~12% faster overall; the slot written when the test
    # fails is at most `ord[N]` and is never read (only `ord[1:n]` is used below).
    @inbounds for i = 1:N
        x = Float64(col[i]) * sgn
        ord[n+1] = (x, i)
        n += ifelse(x > 0.0, 1, 0)
    end
    n == 0 && return out
    _sort_positive!(w, n)

    level = w.level
    parent = w.parent
    sz = w.sz
    head = w.head
    tail = w.tail
    nd_next = w.nd_next
    root_stamp = w.root_stamp
    elem_node = w.elem_node
    nd_parent = w.nd_parent
    nd_size = w.nd_size
    nd_birth = w.nd_birth
    nd_death = w.nd_death
    ptr = adj.ptr
    idx = adj.idx
    nNodes = 0

    p = n                                                     # walk the keys from the top
    while p >= 1
        key = ord[p][1]
        hpow = key^Hp1                              # shared by every node of this
        level += 1                                  # level: `pow` once per level
        q = p
        @inbounds while q >= 1 && ord[q][1] == key  # ties share one level
            q -= 1
        end

        @inbounds for t = (q+1):p                             # activate as singletons
            u = ord[t][2]
            parent[u] = u
            sz[u] = 1
            head[u] = NONODE
            tail[u] = NONODE
        end
        @inbounds for t = (q+1):p                             # merge with active neighbours
            u = ord[t][2]
            ru = _findroot!(parent, u)          # u's root: the same for every neighbour
            for k = (ptr[u]+1):ptr[u+1]
                v = idx[k]
                # `parent[v]` doubles as the activity flag: it is nonzero exactly for
                # elements above the current height. `parent[v] == ru` already proves
                # the merge would be a no-op, so the find can be skipped.
                pv = parent[v]
                if pv != 0 && pv != ru
                    ru = _merge!(parent, sz, head, tail, nd_next, ru, v)
                end
            end
        end
        @inbounds for t = (q+1):p                # one fresh node per changed component
            u = ord[t][2]
            r = _findroot!(parent, u)
            if root_stamp[r] != level
                nNodes += 1
                node = nNodes
                root_stamp[r] = level
                nd_parent[node] = 0
                nd_size[node] = sz[r]
                nd_birth[node] = hpow
                nd_death[node] = 0.0
                o = head[r]                      # every old component of r dies here
                while o != NONODE
                    nd_parent[o] = node
                    nd_death[o] = hpow
                    o = nd_next[o]
                end
                head[r] = node
                tail[r] = node
                nd_next[node] = NONODE
            end
            elem_node[u] = head[r]
        end
        p = q
    end
    w.level = level

    cum = w.cum
    sizeE = w.sizeE
    @inbounds for i = nNodes:-1:1                # a node's parent always has a larger
        c = sizeE[nd_size[i]] * (nd_birth[i] - nd_death[i]) / Hp1   # index, so one
        par = nd_parent[i]                                           # backward pass suffices
        cum[i] = par == 0 ? c : c + cum[par]
    end
    @inbounds for t = 1:n
        u = ord[t][2]
        out[u] += sgn * cum[elem_node[u]]
        parent[u] = 0    # clear the activity flag for exactly what this pass touched
    end
    return out
end

# Merge the component of `v` into the one rooted at `ra` (union by size, splicing the
# merged node lists along in O(1)) and return the surviving root.
@inline function _merge!(parent, sz, head, tail, nd_next, ra, v)
    rb = _findroot!(parent, v)
    ra == rb && return ra
    if sz[ra] < sz[rb]
        ra, rb = rb, ra
    end
    parent[rb] = ra
    sz[ra] += sz[rb]
    hb = head[rb]
    if hb != NONODE
        if head[ra] == NONODE
            head[ra] = hb
            tail[ra] = tail[rb]
        else
            nd_next[tail[ra]] = hb
            tail[ra] = tail[rb]
        end
        head[rb] = NONODE
        tail[rb] = NONODE
    end
    return ra
end

# Union-find find with path halving.
@inline function _findroot!(parent, x)
    @inbounds while parent[x] != x
        px = parent[x]               # parent
        parent[x] = parent[px]       # halve: point x straight at its grandparent
        x = parent[x]
    end
    return x
end
