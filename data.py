from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def generate_er_graph(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    upper = rng.random((n, n)) < p
    upper = np.triu(upper, k=1).astype(np.float32)
    adj = upper + upper.T
    return adj


def generate_two_chains_graph(n: int, k: int) -> np.ndarray:
    if n != 2 * k:
        raise ValueError(f"TwoChains requires n == 2*k, got n={n}, k={k}")
    adj = np.zeros((n, n), dtype=np.float32)
    for start in (0, k):
        for i in range(start, start + k - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
    return adj


def generate_split_chains_graph(n: int, short_len: int) -> np.ndarray:
    """Two disjoint paths partitioning ALL n nodes into components of size
    ``short_len`` and ``n - short_len`` -- an ASYMMETRIC two-chain (Report VI,
    Thread B). With ``short_len = n//2`` this is the balanced ``two_chains`` of
    Reports III--IV; a small ``short_len`` makes one trivial short path beside one
    near-full-length path. Two connected components, NO isolated padding (every node
    lies on a path), max within-component distance ``max(short_len, n-short_len) - 1``.
    No self-loops; node order NOT permuted (permute at the call site).

    The knob is the SPLIT, not the size: it lets us ask whether a graph that is one
    long reach plus a trivial stub (e.g. 4+36 at n=40) is easier than two
    capacity-boundary reaches (e.g. 17+23), holding n fixed -- the Report-IV puzzle."""
    if not 1 <= short_len <= n - 1:
        raise ValueError(f"split requires 1 <= short_len <= n-1, got {short_len} (n={n})")
    adj = np.zeros((n, n), dtype=np.float32)
    for a, b in ((0, short_len), (short_len, n)):     # two contiguous segments -> paths
        for i in range(a, b - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
    return adj


def generate_split_cycles_graph(n: int, short_len: int) -> np.ndarray:
    """Two disjoint CYCLES partitioning ALL n nodes into components of size
    ``short_len`` and ``n - short_len`` -- the cycle analogue of
    ``generate_split_chains_graph`` (Report VIII). Each segment is the same
    contiguous path PLUS its closing edge, so NEITHER component has a degree-1
    endpoint. This isolates whether the model's asymmetric-split behaviour
    (Report VI Thread B / Report VII) depends on the path's open ends acting as
    a distinguished "this component is closed" signal (Report VII sec 4.7's
    far-endpoint-as-source effect) rather than on the split itself: close both
    paths into cycles and the endpoint effect either vanishes (endpoints were
    incidental) or the split can no longer be learned at all (endpoints were
    load-bearing). Requires each segment to have >= 3 nodes (the smallest simple
    cycle). No self-loops; node order NOT permuted (permute at the call site)."""
    if short_len < 3 or n - short_len < 3:
        raise ValueError(f"each cycle needs >= 3 nodes, got short_len={short_len}, "
                          f"n-short_len={n - short_len} (n={n})")
    adj = np.zeros((n, n), dtype=np.float32)
    for a, b in ((0, short_len), (short_len, n)):     # two contiguous segments -> cycles
        for i in range(a, b - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        adj[a, b - 1] = 1.0                           # close the cycle
        adj[b - 1, a] = 1.0
    return adj


def generate_three_way_split_graph(n: int, small_len: int, large_split: int | None = None) -> np.ndarray:
    """Three disjoint paths partitioning ALL n nodes: one SMALL component
    (``small_len`` nodes, meant to sit within capacity and be fully resolvable)
    and two LARGE components (sizes ``large_split`` and
    ``n - small_len - large_split``, meant to each individually exceed capacity)
    that are NOT connected to each other. Default ``large_split`` splits the
    remainder as evenly as possible. Three connected components, no isolated
    padding, no self-loops; node order NOT permuted (permute at the call site).

    Report VII falsification test (Thread B follow-up): if the model resolves
    connectivity by fully identifying a small component and defaulting every
    other pair to ``connected`` (rather than genuinely tracing distance), it
    should wrongly mark the two large, mutually disconnected components as
    connected to each other -- the decisive cross accuracy to read here."""
    if not 1 <= small_len <= n - 2:
        raise ValueError(f"three-way split requires 1 <= small_len <= n-2, got {small_len} (n={n})")
    remainder = n - small_len
    if large_split is None:
        large_split = remainder // 2
    if not 1 <= large_split <= remainder - 1:
        raise ValueError(f"large_split must split the remaining {remainder} nodes into two "
                          f"non-empty paths, got {large_split}")
    adj = np.zeros((n, n), dtype=np.float32)
    bounds = (0, small_len, small_len + large_split, n)  # three contiguous segments -> paths
    for a, b in zip(bounds[:-1], bounds[1:]):
        for i in range(a, b - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
    return adj


def generate_multi_path_split_graph(n: int, sizes: tuple[int, ...]) -> np.ndarray:
    """``len(sizes)`` disjoint paths partitioning ALL n nodes into components of the
    given ordered sizes (``sum(sizes) == n``) -- the general K-component analogue of
    ``generate_split_chains_graph`` (K=2) and ``generate_three_way_split_graph`` (K=3),
    for K=5,6,7,... (Report IX, Thread A.4: does the path-endpoint completion signal
    generalise to a NUMBER of components the training stream never produced -- the
    online path_union stream draws only 1..4 components). Every component size must be
    >=1; components of size 1 are a single isolated node (no internal pairs). No
    self-loops; node order NOT permuted (permute at the call site, as in every other
    generator here)."""
    if any(s < 1 for s in sizes):
        raise ValueError(f"every component size must be >=1, got {sizes}")
    if sum(sizes) != n:
        raise ValueError(f"sizes must sum to n={n}, got {sizes} (sum={sum(sizes)})")
    adj = np.zeros((n, n), dtype=np.float32)
    bounds = [0]
    for s in sizes:
        bounds.append(bounds[-1] + s)
    for a, b in zip(bounds[:-1], bounds[1:]):
        for i in range(a, b - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
    return adj


def generate_stitched_theta_graph(n: int, sizes: tuple[int, ...] = (20, 20, 20),
                                   n_isolated: int = 0) -> np.ndarray:
    """Training-distribution version of eval_n60_multipath_v2.py's Construction B
    (2026-08-31, Edoardo -- train directly on graphs containing a genuine long-range
    redundant-route pair, instead of testing an already-trained plain-disjoint-path model
    OOD on one). Same construction: start from the disjoint ``len(sizes)``-way path split
    (generate_multi_path_split_graph), then add 4 edges joining the middle component's own
    two endpoints to its two neighbours, turning them into a pair of hubs connected by
    THREE routes -- directly through the middle component, and two there-and-back loops
    through the outer two. With sizes=(20,20,20) the three routes are 19/21/21 edges, so the
    hub pair's true shortest-path distance is 19, past the 2*3^L=18 wall, in EVERY sample of
    this family (not just some, unlike a fully randomised route-length design). No self-loops;
    node order NOT permuted (permute at the call site, as in every other generator here).

    ``n_isolated`` (2026-09-01, Edoardo): pads the canvas with that many degree-0 singleton
    nodes when ``sum(sizes) < n`` (e.g. sizes=(19,20,19), n_isolated=2 at n=60 -- the middle
    component still gives a direct hub-hub route of 19 edges, but the two outer components
    now give 20-edge routes instead of 21, i.e. routes 19/20/20 -- plus 2 nodes with no edges
    at all)."""
    if len(sizes) != 3:
        raise ValueError(f"expected exactly 3 components (theta needs a middle one), got {sizes}")
    if sum(sizes) + n_isolated != n:
        raise ValueError(f"sizes+n_isolated must sum to n={n}, got {sizes}+{n_isolated}")
    full_sizes = tuple(sizes) + (1,) * n_isolated
    adj = generate_multi_path_split_graph(n, full_sizes)
    a0, a1 = 0, sizes[0] - 1
    b0, b1 = sizes[0], sizes[0] + sizes[1] - 1
    c0, c1 = sizes[0] + sizes[1], sizes[0] + sizes[1] + sizes[2] - 1
    for u, v in [(a0, b0), (a1, b1), (b0, c0), (b1, c1)]:
        adj[u, v] = adj[v, u] = 1.0
    return adj


def generate_stitched_theta_graph_truncated(n: int, sizes: tuple[int, ...] = (20, 20, 20),
                                             cut: tuple[str, ...] = ("A", "C")) -> np.ndarray:
    """Training-distribution version of eval_n60_multipath_v2.py's Construction C ablation
    (2026-09-01, Edoardo -- train directly on a graph where the hub pair LOOKS like it has
    3 redundant routes but only 1 is real, instead of testing an already-trained
    theta_20_20_20 checkpoint OOD on one). Starts from generate_stitched_theta_graph, then
    removes one INTERNAL edge (strictly between two interior nodes, never touching a hub or
    a stitch endpoint) from each outer route named in ``cut`` ("A" = the route via the first
    component, "C" = via the third) -- so the hub pair (still degree 3 each, still looks
    like a 3-route hub locally) is only actually connected by whichever route(s) are NOT
    cut. Default cuts both outer routes, leaving only the direct through-middle route (dist
    19) functional; the severed outer components become two dead-end arms hanging off each
    hub (the whole graph stays one connected component -- no new isolated nodes). No
    self-loops; node order NOT permuted (permute at the call site, as in every other
    generator here)."""
    if len(sizes) != 3:
        raise ValueError(f"expected exactly 3 components (theta needs a middle one), got {sizes}")
    adj = generate_stitched_theta_graph(n, sizes).copy()
    a0, a1 = 0, sizes[0] - 1
    c0, c1 = sizes[0] + sizes[1], sizes[0] + sizes[1] + sizes[2] - 1
    ranges = {"A": (a0, a1), "C": (c0, c1)}
    for name in cut:
        lo, hi = ranges[name]
        if hi - lo < 2:
            raise ValueError(f"route {name} (size {hi - lo + 1}) too short for an internal cut")
        u, v = (lo + hi) // 2, (lo + hi) // 2 + 1
        if adj[u, v] != 1.0:
            raise AssertionError(f"expected an edge at ({u},{v})")
        adj[u, v] = adj[v, u] = 0.0
    return adj


def generate_hub_truncated_arms_graph(n: int, mid_interior: int, arms: list,
                                       n_isolated: int = 0) -> np.ndarray:
    """Two hub nodes s=0, t=1 joined by a SINGLE direct route through ``mid_interior``
    interior nodes (true distance ``mid_interior+1``). For each ``(a, b)`` pair in
    ``arms``, adds a truncated "false route": a dead-end chain of ``a`` nodes hanging off
    s and a SEPARATE dead-end chain of ``b`` nodes hanging off t -- the two halves are
    never joined, so s and t still look locally like degree-``1+len(arms)`` hubs (as many
    apparent routes as ``len(arms)+1``) but only the direct middle route actually connects
    them (2026-09-01, Edoardo -- generalises generate_stitched_theta_graph_truncated to
    asymmetric arm lengths and an arbitrary number of arms, plus fully isolated padding
    nodes). ``n_isolated`` extra nodes carry no edges at all. Requires
    ``2 + mid_interior + sum(a+b for a,b in arms) + n_isolated == n``. No self-loops; node
    order NOT permuted (permute at the call site, as in every other generator here)."""
    total = 2 + mid_interior + sum(a + b for a, b in arms) + n_isolated
    if total != n:
        raise ValueError(f"sizes must sum to n={n}, got {total} "
                          f"(mid_interior={mid_interior}, arms={arms}, n_isolated={n_isolated})")
    adj = np.zeros((n, n), dtype=np.float32)
    s, t = 0, 1
    cur = 2
    prev = s
    for _ in range(mid_interior):
        adj[prev, cur] = adj[cur, prev] = 1.0
        prev = cur; cur += 1
    adj[prev, t] = adj[t, prev] = 1.0
    for a, b in arms:
        prev = s
        for _ in range(a):
            adj[prev, cur] = adj[cur, prev] = 1.0
            prev = cur; cur += 1
        prev = t
        for _ in range(b):
            adj[prev, cur] = adj[cur, prev] = 1.0
            prev = cur; cur += 1
    return adj


def generate_multi_cycle_split_graph(n: int, sizes: tuple[int, ...]) -> np.ndarray:
    """``len(sizes)`` disjoint CYCLES partitioning ALL n nodes into components of the given
    ordered sizes (``sum(sizes) == n``) -- the cycle analogue of
    ``generate_multi_path_split_graph``, generalising ``generate_split_cycles_graph`` (K=2) to
    K=3,4,... . Every component size must be >=3 (the smallest simple cycle). No self-loops;
    node order NOT permuted (permute at the call site, as in every other generator here)."""
    if any(s < 3 for s in sizes):
        raise ValueError(f"every cycle needs >=3 nodes, got {sizes}")
    if sum(sizes) != n:
        raise ValueError(f"sizes must sum to n={n}, got {sizes} (sum={sum(sizes)})")
    adj = np.zeros((n, n), dtype=np.float32)
    bounds = [0]
    for s in sizes:
        bounds.append(bounds[-1] + s)
    for a, b in zip(bounds[:-1], bounds[1:]):
        for i in range(a, b - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        adj[a, b - 1] = 1.0
        adj[b - 1, a] = 1.0
    return adj


def generate_split_cliques_asym_graph(n: int, short_len: int) -> np.ndarray:
    """Two disjoint COMPLETE cliques partitioning ALL n nodes into components of size
    ``short_len`` and ``n - short_len`` -- the clique analogue of
    ``generate_split_chains_graph``/``generate_split_cycles_graph`` (Report IX, controlled-
    distribution battery). Unlike ``generate_split_cliques_graph`` (fixed EQUAL clique size,
    isolated padding for the remainder), this covers every node and allows an asymmetric
    split, so the (a, n-a) sweep used throughout this project applies unchanged. Every node's
    DEGREE immediately reveals its component's size (degree = component size - 1) -- the
    control this family is FOR: if imbalanced splits are solved easily here, the model can be
    using a trivial global degree signature, not anything endpoint- or landmark-specific.
    Requires each clique to have >=1 node (a size-1 clique is an isolated node, no internal
    edges). No self-loops; node order NOT permuted (permute at the call site)."""
    if not 1 <= short_len <= n - 1:
        raise ValueError(f"split requires 1 <= short_len <= n-1, got {short_len} (n={n})")
    adj = np.zeros((n, n), dtype=np.float32)
    for a, b in ((0, short_len), (short_len, n)):
        idx = np.arange(a, b)
        adj[np.ix_(idx, idx)] = 1.0
    np.fill_diagonal(adj, 0.0)
    return adj


def generate_chorded_cycles_graph(n: int, short_len: int) -> np.ndarray:
    """Two disjoint CYCLES partitioning ALL n nodes into components of size ``short_len``
    and ``n - short_len``, EACH with one additional CHORD edge connecting two
    (approximately) opposite nodes within that cycle -- Report IX's middle rung between
    plain cycles (no distinguishing landmark at all) and paths (two degree-1 endpoints):
    exactly 2 nodes per cycle get degree 3 (a landmark that breaks the cycle's rotational
    symmetry) while every other node keeps degree 2 and neither cycle has an open endpoint.
    Isolates whether ANY symmetry-breaking landmark is enough for the completion signal, or
    whether path ENDPOINTS specifically are needed. Each cycle needs >=4 nodes (the smallest
    cycle with a non-adjacent pair to chord). No self-loops; node order NOT permuted (permute
    at the call site)."""
    if short_len < 4 or n - short_len < 4:
        raise ValueError(f"each chorded cycle needs >=4 nodes, got short_len={short_len}, "
                          f"n-short_len={n - short_len} (n={n})")
    adj = np.zeros((n, n), dtype=np.float32)
    for a, b in ((0, short_len), (short_len, n)):
        size = b - a
        for i in range(a, b - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        adj[a, b - 1] = 1.0                  # close the cycle
        adj[b - 1, a] = 1.0
        chord_j = a + size // 2              # ~opposite node, always non-adjacent for size>=4
        adj[a, chord_j] = 1.0
        adj[chord_j, a] = 1.0
    return adj


def generate_split_regular_graph(n: int, d: int, short_len: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """Two disjoint, CONNECTED, ``d``-regular random graphs partitioning ALL n nodes into
    components of size ``short_len`` and ``n - short_len`` -- Report IX's cleanest control:
    every node has the SAME degree ``d`` everywhere (no endpoints, no chords, no degree
    signature of component size at all), so if imbalanced splits are still solved more easily
    than balanced ones, the effect cannot be attributed to any local landmark or global degree
    cue -- it would have to come from the component-SIZE asymmetry itself. Uses
    ``networkx.random_regular_graph`` per block (retried on the rare disconnected draw --
    connectivity is not guaranteed by construction for d>=3). Each block size must be >d and
    ``d*size`` even (the standard random-regular-graph feasibility condition); with ``d=3``
    (this project's default) both ``short_len`` and ``n-short_len`` must be even and >=4. No
    self-loops; node order NOT permuted (permute at the call site)."""
    import networkx as nx
    if not 1 <= short_len <= n - 1:
        raise ValueError(f"split requires 1 <= short_len <= n-1, got {short_len} (n={n})")
    adj = np.zeros((n, n), dtype=np.float32)
    for a, b in ((0, short_len), (short_len, n)):
        size = b - a
        if size <= d or (d * size) % 2 != 0:
            raise ValueError(f"infeasible d-regular block: size={size}, d={d} "
                              f"(need size>d and d*size even)")
        for _attempt in range(50):
            g = nx.random_regular_graph(d, size, seed=int(rng.integers(0, 2**31 - 1)))
            if nx.is_connected(g):
                break
        else:
            raise RuntimeError(f"could not draw a CONNECTED {d}-regular graph on {size} "
                               f"nodes in 50 attempts")
        block_adj = nx.to_numpy_array(g, nodelist=sorted(g.nodes()))
        idx = np.arange(a, b)
        adj[np.ix_(idx, idx)] = block_adj
    return adj


def generate_two_cliques_graph(n: int, k: int) -> np.ndarray:
    if n != 2 * k:
        raise ValueError(f"TwoCliques requires n == 2*k, got n={n}, k={k}")
    adj = np.zeros((n, n), dtype=np.float32)
    for start in (0, k):
        idx = np.arange(start, start + k)
        adj[np.ix_(idx, idx)] = 1.0
        np.fill_diagonal(adj[start : start + k, start : start + k], 0.0)
    return adj


def generate_bridged_cliques_graph(n: int, rng: np.random.Generator | None = None,
                                   clique_size: int | None = None,
                                   bridge_width: int = 1) -> np.ndarray:
    """Two cliques joined by a bridge -> ONE connected component.

    The advisor's DFS probe (Report V). Differs from generate_split_cliques_graph
    (two cliques, NO bridge -> two components) by EXACTLY the bridge edges, and those
    edges create a connection at shortest-path distance <= 3 (clique A -> bridge -> bridge
    -> clique B). So matrix powering (A^{3^L}, capacity 9) detects it trivially at
    any clique size, whereas a step/visit-bounded DFS must traverse a whole dense
    clique before reaching the far one and so fails once a clique exceeds its budget.

    ``clique_size`` c: each clique has c nodes (2c <= n); the remaining n-2c nodes
    are isolated padding. Default c = n//2 (no padding). ``bridge_width`` w (>=1, <= c):
    the number of edges joining the two cliques (Report VI, Thread C.2: a THICKER
    bridge = redundant hand-off routes, like Thread A's parallel paths, while the
    cross distance stays <= 3 within the matrix-power capacity). w=1 is the Report-V
    single bridge: clique A's last node (c-1) to clique B's first node (c). For w>1 we
    add w disjoint edges (c-1-j, c+j) for j=0..w-1. No self-loops; node order NOT
    permuted (permute at the call site). ``rng`` is taken for a uniform generator
    signature but unused (the graph is fixed given c and w)."""
    c = n // 2 if clique_size is None else max(2, min(int(clique_size), n // 2))
    w = max(1, min(int(bridge_width), c))
    adj = np.zeros((n, n), dtype=np.float32)
    for idx in (np.arange(0, c), np.arange(c, 2 * c)):
        adj[np.ix_(idx, idx)] = 1.0
    np.fill_diagonal(adj, 0.0)
    for j in range(w):                       # w bridge edges between clique A and B
        adj[c - 1 - j, c + j] = adj[c + j, c - 1 - j] = 1.0
    return adj


def generate_split_cliques_graph(n: int, rng: np.random.Generator | None = None,
                                 clique_size: int | None = None) -> np.ndarray:
    """Two cliques, NO bridge -> TWO components (the negative label of the
    bridged-cliques probe, Report V). ``clique_size`` c as in
    generate_bridged_cliques_graph; with c=n//2 this equals
    generate_two_cliques_graph(n, n//2). Remaining n-2c nodes are isolated."""
    c = n // 2 if clique_size is None else max(2, min(int(clique_size), n // 2))
    adj = np.zeros((n, n), dtype=np.float32)
    for idx in (np.arange(0, c), np.arange(c, 2 * c)):
        adj[np.ix_(idx, idx)] = 1.0
    np.fill_diagonal(adj, 0.0)
    return adj


def generate_bridged_blocks_graph(n: int, rng: np.random.Generator | None = None,
                                  clique_size: int | None = None,
                                  bridged: bool = True, p_in: float = 0.6,
                                  bridge_width: int = 1) -> np.ndarray:
    """Two internally-DENSE ER blocks of size c joined by a SINGLE bridge edge
    (``bridged=True`` -> one component) or not (``bridged=False`` -> two components).

    The HELD-OUT OOD analogue of the bridged cliques (Report V): the SAME challenge
    -- carry one bridge edge across a dense region of c nodes -- but each region is a
    dense ER block, NOT a complete clique, so a model that merely memorised the clique
    family cannot match by pattern. Each block is built as a random spanning path over
    its c nodes (guarantees connectivity) plus extra edges at probability ``p_in``; we
    keep p_in HIGH so the within-block distance stays ~2 and the cross-block distance
    stays well within the matrix-power capacity 3^L=9 (so the test isolates
    propagation, not distance).

    ``clique_size`` c: each block has c nodes (2c <= n), the rest isolated padding;
    blocks sit at [0,c) and [c,2c). ``bridge_width`` w (>=1, <= c): the number of
    bridge edges joining the two blocks (Report VI, Thread C.2 thick bridges); w=1 is
    the single bridge of Report V (node c-1 to node c). No self-loops; node order NOT
    permuted (permute at the call site). ``rng`` is REQUIRED (the blocks are random);
    a default generator is created if omitted."""
    if rng is None:
        rng = np.random.default_rng()
    c = n // 2 if clique_size is None else max(2, min(int(clique_size), n // 2))
    w = max(1, min(int(bridge_width), c))
    adj = np.zeros((n, n), dtype=np.float32)
    for base in (0, c):
        idx = np.arange(base, base + c)
        order = rng.permutation(idx)
        for i in range(c - 1):                       # spanning path -> connected
            u, v = int(order[i]), int(order[i + 1])
            adj[u, v] = adj[v, u] = 1.0
        for ii in range(c):                          # densify
            for jj in range(ii + 1, c):
                u, v = int(idx[ii]), int(idx[jj])
                if adj[u, v] == 0.0 and rng.random() < p_in:
                    adj[u, v] = adj[v, u] = 1.0
    if bridged:
        for j in range(w):                           # w bridge edges between blocks
            adj[c - 1 - j, c + j] = adj[c + j, c - 1 - j] = 1.0
    return adj


def generate_clique_chain_graph(n: int, clique_size: int, n_cliques: int,
                                rng: np.random.Generator | None = None,
                                bridge_width: int = 1, block: str = "clique",
                                p_in: float = 0.6,
                                broken_link: int | None = None) -> np.ndarray:
    """A CHAIN of dense blocks: ``n_cliques`` blocks of ``clique_size`` nodes in a row,
    each adjacent pair joined by a bridge -> ONE connected component (Report VI, Thread
    C.1: does a learned single bridge COMPOSE across repeated hand-offs?).

    The iterated analogue of ``generate_bridged_cliques_graph``: instead of two cliques
    and one bridge it is K cliques and K-1 bridges, so the model must find, at each
    block, the node that carries the link to the next block -- a forced hand-off,
    repeated. Block i occupies indices [i*c, (i+1)*c); the bridge between block i and
    i+1 joins block i's last ``bridge_width`` nodes to block i+1's first ``bridge_width``
    nodes (so the cross distance between adjacent blocks stays <= 3, inside the
    matrix-power capacity 9 -- but the END-TO-END distance grows ~2 hops per added block,
    so the caller MUST check the diameter stays <= 9; see eval feasibility).

    ``block="clique"``: each block is a complete clique (intra distance 1). ``block="er"``:
    each block is a dense ER block (random spanning path + extra edges at prob ``p_in``),
    the held-out OOD analogue of Report V sec 5.6. ``broken_link`` l (optional): DROP the
    bridge between block l and l+1 -> TWO components (blocks 0..l and l+1..K-1) -- the
    negative used to check the model is really reading the bridges, not predicting
    all-connected. The remaining n - K*c nodes are isolated padding. No self-loops; node
    order NOT permuted (permute at the call site)."""
    c, K = int(clique_size), int(n_cliques)
    if c < 2 or K < 1:
        raise ValueError(f"need clique_size>=2 and n_cliques>=1, got c={c}, K={K}")
    if K * c > n:
        raise ValueError(f"need K*c={K*c} <= n={n} for the chain to fit")
    w = max(1, min(int(bridge_width), c))
    if rng is None:
        rng = np.random.default_rng()
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(K):                                # build the K blocks
        idx = np.arange(i * c, (i + 1) * c)
        if block == "clique":
            adj[np.ix_(idx, idx)] = 1.0
            np.fill_diagonal(adj[i * c:(i + 1) * c, i * c:(i + 1) * c], 0.0)
        elif block == "er":
            order = rng.permutation(idx)
            for k in range(c - 1):                    # spanning path -> connected
                u, v = int(order[k]), int(order[k + 1])
                adj[u, v] = adj[v, u] = 1.0
            for ii in range(c):                       # densify
                for jj in range(ii + 1, c):
                    u, v = int(idx[ii]), int(idx[jj])
                    if adj[u, v] == 0.0 and rng.random() < p_in:
                        adj[u, v] = adj[v, u] = 1.0
        else:
            raise ValueError(f"unknown block kind {block!r}")
    for i in range(K - 1):                             # K-1 bridges
        if broken_link is not None and i == int(broken_link):
            continue                                   # drop this bridge -> two components
        a_last = i * c + c - 1                          # last node of block i
        b_first = (i + 1) * c                           # first node of block i+1
        for j in range(w):
            adj[a_last - j, b_first + j] = adj[b_first + j, a_last - j] = 1.0
    return adj


def generate_one_chain_graph(n: int) -> np.ndarray:
    """Single path (chain) over all n nodes: 0-1-2-...-(n-1). One connected
    component. Used as the negative ("1 chain") class in the components task."""
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj


def generate_directed_chain_graph(n: int) -> np.ndarray:
    """DIRECTED analogue of generate_one_chain_graph, for the directed-reachability /
    "genuine reasoning chain" task (2026-08-28, Edoardo -- testing whether the multipath-
    redundancy connectivity finding generalises to directed implication-style chains,
    A=>B=>C..., in the spirit of Abbe et al.'s syllogism-composition task). A single directed
    Hamiltonian path 0->1->...->(n-1): edge i->i+1 only, no reverse edge. Node identity is
    randomised the same way as every other family (sample_family's permutation step applies
    to both rows and columns, which correctly preserves directed edges under relabelling)."""
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
    return adj


def generate_path_union_graph(n: int, rng: np.random.Generator,
                              max_paths: int = 4, min_paths: int = 1) -> np.ndarray:
    """Disjoint union of min_paths..max_paths paths that partition all n nodes.

    Designed to exercise long-range reachability: with k=1 the graph is a single
    path of n nodes (shortest-path distances up to n-1), while k>1 introduces
    genuine cross-component disconnections so the model cannot trivially predict
    "all connected". Node order is NOT permuted here (do it at the call site).

    ``min_paths`` (Report IX, readout-only fine-tuning experiment): raising it above
    the default 1 restricts the stream to graphs with at least that many components
    -- e.g. min_paths=3 gives a "3-or-more-paths" distribution genuinely disjoint
    from the k in {1,2} cases a 2-chain-trained model already saw."""
    if not 1 <= min_paths <= max_paths:
        raise ValueError(f"require 1 <= min_paths <= max_paths, got min_paths={min_paths}, "
                         f"max_paths={max_paths}")
    k = int(rng.integers(min_paths, max_paths + 1))
    if k == 1:
        bounds = [0, n]
    else:
        cuts = sorted(rng.choice(np.arange(1, n), size=k - 1, replace=False).tolist())
        bounds = [0] + cuts + [n]
    adj = np.zeros((n, n), dtype=np.float32)
    for a, b in zip(bounds[:-1], bounds[1:]):
        for i in range(a, b - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
    return adj


def generate_parallel_paths_graph(n: int, n_paths: int, path_len: int) -> np.ndarray:
    """Two terminal nodes (0 and 1) joined by ``n_paths`` internally-disjoint paths,
    each of ``path_len`` edges. The shortest-path distance between the terminals is
    ``path_len`` for ANY n_paths, but the effective resistance between them is
    ``path_len / n_paths`` -- so this isolates distance (fixed) from the number of
    paths (varied). Remaining nodes are isolated padding. The terminals + internal
    nodes form one connected component. No self-loops; node order NOT permuted.
    Requires 2 + n_paths*(path_len-1) <= n."""
    need = 2 + n_paths * (path_len - 1)
    if need > n:
        raise ValueError(f"need {need} nodes for n_paths={n_paths}, path_len={path_len}, "
                         f"but n={n}")
    adj = np.zeros((n, n), dtype=np.float32)
    s, t = 0, 1
    cur = 2
    for _ in range(n_paths):
        prev = s
        for _ in range(path_len - 1):
            adj[prev, cur] = adj[cur, prev] = 1.0
            prev = cur
            cur += 1
        adj[prev, t] = adj[t, prev] = 1.0
    return adj


def generate_multipath_graph(n: int, n_full: int, path_len: int | list[int],
                             rng: np.random.Generator, n_trunc: int = 0,
                             term_deg: int = 4, trunc_len: int | None = None,
                             fill: bool = True):
    """Report VI multipath family (the confound-free parallel-paths construction of
    Report IV, generalised + carrying its own structure for per-path analysis, and
    supporting TRUNCATED dead-end routes for Thread A3).

    Two terminals ``s=0``, ``t=1`` are joined by ``n_full`` internally-disjoint paths.
    ``path_len`` is either a single int (every route the same length, the original
    behaviour: ``dist(s,t)=path_len`` for any ``n_full``) or a list of ``n_full`` ints
    (Report X: routes of DIFFERENT lengths, e.g. to match a disjoint K-way split's
    component sizes as closely as possible) -- ``dist(s,t)`` is then ``min(path_len)``.
    Optionally ``n_trunc`` DEAD-END paths of ``trunc_len`` edges hang off ``s`` only
    (they never reach ``t``): these are the truncated routes. Each terminal is padded
    with leaf nodes to keep its degree near ``term_deg`` (so a terminal does not become
    a higher-degree hub as the number of routes grows), and the remaining canvas is
    wired into one sparse filler path (a separate component) so the mean degree stays
    ~2 like the sparse training graphs.

    ``s`` and ``t`` are in the same component iff ``n_full >= 1``. Returns
    ``(adj_no_loops, meta)`` on UNPERMUTED indices (permute with
    :func:`permute_with_meta`), or ``None`` if the construction does not fit in ``n``.
    ``meta`` carries ``s, t, full_paths, trunc_paths, leaves, filler, n_full, n_trunc,
    path_len`` (every path is the list of its INTERNAL node indices; ``path_len`` in
    ``meta`` is always the per-route list, even if a single int was passed in)."""
    s, t = 0, 1
    path_lens = [path_len] * n_full if isinstance(path_len, int) else list(path_len)
    if len(path_lens) != n_full:
        raise ValueError(f"path_len list must have n_full={n_full} entries, got {len(path_lens)}")
    if trunc_len is None:
        trunc_len = max(1, path_lens[0] - 1)
    leaves_s = max(0, term_deg - n_full - n_trunc)
    leaves_t = max(0, term_deg - n_full)
    need = (2 + sum(pl - 1 for pl in path_lens) + n_trunc * trunc_len + leaves_s + leaves_t)
    if need > n:
        return None
    adj = np.zeros((n, n), dtype=np.float32)
    cur = 2
    full_paths: list[list[int]] = []
    trunc_paths: list[list[int]] = []
    for pl in path_lens:
        prev = s; nodes: list[int] = []
        for _ in range(pl - 1):
            adj[prev, cur] = adj[cur, prev] = 1.0
            nodes.append(cur); prev = cur; cur += 1
        adj[prev, t] = adj[t, prev] = 1.0
        full_paths.append(nodes)
    for _ in range(n_trunc):
        prev = s; nodes = []
        for _ in range(trunc_len):
            adj[prev, cur] = adj[cur, prev] = 1.0
            nodes.append(cur); prev = cur; cur += 1
        trunc_paths.append(nodes)            # dead-end: no edge to t
    leaves: list[int] = []
    for _ in range(leaves_s):
        adj[s, cur] = adj[cur, s] = 1.0; leaves.append(cur); cur += 1
    for _ in range(leaves_t):
        adj[t, cur] = adj[cur, t] = 1.0; leaves.append(cur); cur += 1
    filler = list(range(cur, n))
    if fill:
        for i in range(len(filler) - 1):
            adj[filler[i], filler[i + 1]] = adj[filler[i + 1], filler[i]] = 1.0
    meta = {"s": s, "t": t, "full_paths": full_paths, "trunc_paths": trunc_paths,
            "leaves": leaves, "filler": filler, "n_full": n_full, "n_trunc": n_trunc,
            "path_len": path_lens}
    return adj, meta


def permute_with_meta(adj: np.ndarray, meta: dict, rng: np.random.Generator):
    """Apply a random node permutation to a multipath graph and remap its meta
    indices, so the terminals/paths can still be located after shuffling."""
    n = adj.shape[0]
    p = rng.permutation(n); inv = np.argsort(p)
    a2 = adj[np.ix_(p, p)]
    rm = lambda xs: [int(inv[x]) for x in xs]
    m2 = {"s": int(inv[meta["s"]]), "t": int(inv[meta["t"]]),
          "full_paths": [rm(pp) for pp in meta["full_paths"]],
          "trunc_paths": [rm(pp) for pp in meta["trunc_paths"]],
          "leaves": rm(meta["leaves"]), "filler": rm(meta["filler"]),
          "n_full": meta["n_full"], "n_trunc": meta["n_trunc"],
          "path_len": meta["path_len"]}
    return a2, m2


def generate_chain_plus_graph(n: int, rng: np.random.Generator) -> np.ndarray:
    """A long chain plus a small separate component. Designed to expose the 3^L wall
    even at small n: the long path gives within-component distances well past 9, while
    the second component forces a real disconnection (so a trivial "predict connected"
    is wrong on the cross-component pairs). Long path of n-s nodes + short path of s
    nodes, s ~ U[3, n//3]; two components, max within-distance n-s-1. No self-loops;
    node order NOT permuted."""
    s = int(rng.integers(3, max(4, n // 3 + 1)))
    long_len = n - s
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(long_len - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    for i in range(long_len, n - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    return adj


def generate_one_cycle_graph(n: int) -> np.ndarray:
    """Single cycle C_n over all n nodes (path + closing edge). One connected
    component, every node has degree 2, diameter floor(n/2)."""
    if n < 3:
        raise ValueError(f"a cycle needs n >= 3, got n={n}")
    adj = generate_one_chain_graph(n)
    adj[0, n - 1] = 1.0
    adj[n - 1, 0] = 1.0
    return adj


def generate_two_cycles_graph(n: int, k: int) -> np.ndarray:
    """Two disjoint cycles C_k. Requires n == 2*k and k >= 3. Two connected
    components, each of diameter floor(k/2)."""
    if n != 2 * k:
        raise ValueError(f"TwoCycles requires n == 2*k, got n={n}, k={k}")
    if k < 3:
        raise ValueError(f"each cycle needs k >= 3, got k={k}")
    adj = np.zeros((n, n), dtype=np.float32)
    for start in (0, k):
        for i in range(start, start + k - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        # close each cycle
        adj[start, start + k - 1] = 1.0
        adj[start + k - 1, start] = 1.0
    return adj


def generate_blocks_graph(n: int, rng: np.random.Generator, kind: str = "er",
                          max_blocks: int = 4, p: float = 0.3) -> np.ndarray:
    """Disjoint union of 1..max_blocks internally-connected blocks partitioning all
    n nodes (the dense analogue of ``generate_path_union_graph``).

    ``kind="clique"``: each block is a complete graph (intra-block diameter 1,
    large spectral gap). ``kind="er"``: each block is a *connected* ER graph,
    built as a random spanning path over the block (guarantees connectivity) plus
    extra ER edges at probability ``p`` (variable density, variable diameter/gap).
    Blocks of size 1 stay isolated nodes (own component). Node order is NOT
    permuted here; no self-loops."""
    k = int(rng.integers(1, max_blocks + 1))
    if k == 1:
        bounds = [0, n]
    else:
        cuts = sorted(rng.choice(np.arange(1, n), size=k - 1, replace=False).tolist())
        bounds = [0] + cuts + [n]
    adj = np.zeros((n, n), dtype=np.float32)
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = np.arange(a, b)
        m = idx.size
        if m <= 1:
            continue
        if kind == "clique":
            adj[np.ix_(idx, idx)] = 1.0
            np.fill_diagonal(adj[a:b, a:b], 0.0)
        elif kind == "er":
            order = rng.permutation(idx)
            for i in range(m - 1):
                u, v = int(order[i]), int(order[i + 1])
                adj[u, v] = adj[v, u] = 1.0
            if p > 0.0:
                for ii in range(m):
                    for jj in range(ii + 1, m):
                        u, v = int(idx[ii]), int(idx[jj])
                        if adj[u, v] == 0.0 and rng.random() < p:
                            adj[u, v] = adj[v, u] = 1.0
        else:
            raise ValueError(f"unknown block kind {kind!r}")
    return adj


def generate_barbell_graph(n: int, rng: np.random.Generator,
                           clique_size: int | None = None) -> np.ndarray:
    """Two cliques joined by a path -- the textbook small-spectral-gap graph
    (one connected component, moderate diameter but a hard bottleneck). No
    self-loops; node order NOT permuted.

    ``clique_size=None`` -> ~n/3 (the fixed barbell family). Passing an int sets the
    end-clique size, hence the bridge length and the spectral gap: this is the clean
    knob for the spectral-gap experiment (small cliques -> long bridge -> tiny gap;
    large cliques -> short bridge -> larger gap), varied *at fixed structure*. The
    'barbell_var' eval family samples clique_size to sweep the gap continuously."""
    if clique_size is None:
        c = max(2, n // 3)
    else:
        c = max(2, min(int(clique_size), n // 2))
    if 2 * c > n:
        c = n // 2
    adj = np.zeros((n, n), dtype=np.float32)
    for s in (0, n - c):
        idx = np.arange(s, s + c)
        adj[np.ix_(idx, idx)] = 1.0
        np.fill_diagonal(adj[s : s + c, s : s + c], 0.0)
    path_nodes = list(range(c - 1, n - c + 1))   # last of clique A ... first of clique B
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        adj[u, v] = adj[v, u] = 1.0
    return adj


def generate_random_regular_graph(n: int, rng: np.random.Generator,
                                  degree: int = 3) -> np.ndarray:
    """Approximate random d-regular graph -- an expander: small diameter, *large*
    spectral gap (the opposite of a path/cycle at the same n). Built as a random
    Hamiltonian cycle (guarantees connectivity, degree 2) plus extra random
    matchings up to the target degree. No self-loops; node order NOT permuted."""
    adj = np.zeros((n, n), dtype=np.float32)
    order = rng.permutation(n)
    for i in range(n):
        u, v = int(order[i]), int(order[(i + 1) % n])
        adj[u, v] = adj[v, u] = 1.0
    for _ in range(max(0, degree - 2)):
        perm = rng.permutation(n)
        for i in range(0, n - 1, 2):
            u, v = int(perm[i]), int(perm[i + 1])
            if u != v and adj[u, v] == 0.0:
                adj[u, v] = adj[v, u] = 1.0
    return adj


def add_self_loops(adj: np.ndarray) -> np.ndarray:
    out = adj.copy().astype(np.float32)
    np.fill_diagonal(out, 1.0)
    return out


def _bfs_distances(adj_no_loops: np.ndarray, source: int) -> np.ndarray:
    n = adj_no_loops.shape[0]
    dist = -np.ones(n, dtype=np.int64)
    dist[source] = 0
    q: list[int] = [source]
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        neighbors = np.where(adj_no_loops[u] > 0)[0]
        for v in neighbors.tolist():
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def compute_all_pairs_shortest_paths(adj_no_loops: np.ndarray) -> np.ndarray:
    """All-pairs shortest paths on an unweighted undirected graph.
    Unreachable pairs are encoded as ``-1`` (same convention as before).
    Uses scipy.sparse.csgraph.shortest_path, which is a C-optimised BFS:
    ~50-100x faster than the previous Python implementation for n ≈ 40.
    Critical for rejection-sampling rates < 50% (D ≤ 9, D ≤ 7) where the
    diameter filter is called many times per accepted graph."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    g = csr_matrix(adj_no_loops.astype(bool).astype(np.int8))
    d = shortest_path(g, method="auto", unweighted=True, directed=False)
    d = np.where(np.isfinite(d), d, -1.0)   # avoid inf -> int64 cast warning
    return d.astype(np.int64)


def connected_components(adj_no_loops: np.ndarray) -> list[list[int]]:
    n = adj_no_loops.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps: list[list[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        comp: list[int] = []
        q = [i]
        seen[i] = True
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            comp.append(u)
            neighbors = np.where(adj_no_loops[u] > 0)[0]
            for v in neighbors.tolist():
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        comps.append(comp)
    return comps


def compute_connectivity_matrix(adj_no_loops: np.ndarray) -> np.ndarray:
    n = adj_no_loops.shape[0]
    conn = np.zeros((n, n), dtype=np.float32)
    comps = connected_components(adj_no_loops)
    for comp in comps:
        idx = np.array(comp, dtype=np.int64)
        conn[np.ix_(idx, idx)] = 1.0
    return conn


def compute_directed_reachability_matrix(adj_no_loops: np.ndarray) -> np.ndarray:
    """Directed reachability: reach[i,j]=1 iff j is reachable from i by following directed
    edges forward only (reach[i,i]=1 always, matching compute_connectivity_matrix's own
    diagonal convention). Reuses _bfs_distances as-is -- it already only follows outgoing
    edges (row u = u's out-neighbours), so it is directed-correct without any change, whether
    adj_no_loops is symmetric or not. NOT symmetric in general: reach[i,j] != reach[j,i] for a
    directed graph, unlike compute_connectivity_matrix. Used only by the "directed_chain"
    family (2026-08-28); every other family stays on compute_connectivity_matrix."""
    n = adj_no_loops.shape[0]
    reach = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        dist = _bfs_distances(adj_no_loops, i)
        reach[i] = (dist >= 0).astype(np.float32)
    return reach


def compute_graph_diameter(adj_no_loops: np.ndarray) -> int:
    d = compute_all_pairs_shortest_paths(adj_no_loops)
    finite = d[d >= 0]
    if finite.size == 0:
        return 0
    return int(finite.max())


def compute_spectral_gap(adj_no_loops: np.ndarray, tol: float = 1e-6) -> float:
    """Smallest *positive* eigenvalue of the normalised Laplacian
    ``L = I - D^{-1/2} A D^{-1/2}``, restricted to non-isolated nodes.

    The multiplicity of the zero eigenvalue equals the number of connected
    components, so the first eigenvalue above ``tol`` measures how hard it is to
    mix *within* components -- small for paths/cycles/barbell (long, bottlenecked)
    and large for cliques/expanders. This is the spectral analogue of the diameter
    and lets us ask whether the model's error tracks the gap rather than the
    diameter. Returns ``0.0`` for an edgeless graph."""
    deg = adj_no_loops.sum(axis=1)
    nz = np.where(deg > 0)[0]
    if nz.size == 0:
        return 0.0
    a = adj_no_loops[np.ix_(nz, nz)].astype(np.float64)
    dinv = 1.0 / np.sqrt(deg[nz].astype(np.float64))
    lap = np.eye(nz.size) - (dinv[:, None] * a * dinv[None, :])
    lap = 0.5 * (lap + lap.T)
    w = np.linalg.eigvalsh(lap)
    pos = w[w > tol]
    if pos.size == 0:
        return 0.0
    return float(pos.min())


def effective_resistance(adj_no_loops: np.ndarray) -> np.ndarray:
    """Effective resistance R(i,j) between every pair, each edge = a unit resistor:
    R(i,j) = (e_i-e_j)^T L^+ (e_i-e_j), with L = D - A the combinatorial Laplacian and
    L^+ its Moore-Penrose pseudo-inverse. Low R = many short parallel paths (easy to
    mix), high R = a single long path / a bottleneck. Computed per connected component;
    pairs in different components get np.inf. Returns an (n, n) float array.

    Note on scale: R has units that depend on the graph's edge count, so absolute
    values are NOT comparable across graphs of different density -- compare ranks
    within a graph, or use diffusion_reach (a probability in [0,1]) across graphs."""
    n = adj_no_loops.shape[0]
    R = np.full((n, n), np.inf, dtype=np.float64)
    for comp in connected_components(adj_no_loops):
        idx = np.array(sorted(comp))
        if idx.size == 1:
            R[idx[0], idx[0]] = 0.0
            continue
        a = adj_no_loops[np.ix_(idx, idx)].astype(np.float64)
        lap = np.diag(a.sum(1)) - a
        lp = np.linalg.pinv(lap)
        d = np.diag(lp)
        R[np.ix_(idx, idx)] = d[:, None] + d[None, :] - 2.0 * lp
    return R


def diffusion_reach(adj_no_loops: np.ndarray, n_steps: int) -> np.ndarray:
    """(P^L)_{ij}: probability that an L-step random walk with self-loops started at i
    sits at j, with P = D^{-1}(A+I) row-stochastic and L = n_steps. High value = i can
    influence j within L message-passing steps, i.e. an L-layer model can mix them.
    A probability in [0,1], hence comparable across graphs. Returns an (n, n) array."""
    n = adj_no_loops.shape[0]
    m = adj_no_loops.astype(np.float64) + np.eye(n)
    p = m / m.sum(1, keepdims=True)
    return np.linalg.matrix_power(p, n_steps)


@dataclass
class DatasetConfig:
    mode: str
    n: int = 20
    p: float = 0.08
    size: int = 1000
    k: int = 10
    seed: int = 0
    max_diameter: int | None = None
    max_attempts: int = 200000


class GraphMatrixDataset(Dataset[dict[str, Any]]):
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.samples: list[dict[str, Any]] = []
        self._generate()

    def _generate_one(self, rng: np.random.Generator) -> np.ndarray:
        if self.config.mode == "er":
            adj = generate_er_graph(self.config.n, self.config.p, rng)
        elif self.config.mode == "two_chains":
            adj = generate_two_chains_graph(self.config.n, self.config.k)
        elif self.config.mode == "two_cliques":
            adj = generate_two_cliques_graph(self.config.n, self.config.k)
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

        if self.config.mode in {"two_chains", "two_cliques"}:
            perm = rng.permutation(self.config.n)
            adj = adj[np.ix_(perm, perm)]

        return adj

    def _generate(self) -> None:
        rng = np.random.default_rng(self.config.seed)
        attempts = 0
        while len(self.samples) < self.config.size:
            attempts += 1
            if attempts > self.config.max_attempts:
                raise RuntimeError(
                    f"Could not generate enough samples for mode={self.config.mode} "
                    f"with max_diameter={self.config.max_diameter}. "
                    f"Generated {len(self.samples)}/{self.config.size} in {attempts} attempts."
                )
            adj_no_loops = self._generate_one(rng)
            diameter = compute_graph_diameter(adj_no_loops)
            if self.config.max_diameter is not None and diameter > self.config.max_diameter:
                continue
            adj = add_self_loops(adj_no_loops)
            target = compute_connectivity_matrix(adj_no_loops)
            dist = compute_all_pairs_shortest_paths(adj_no_loops)
            self.samples.append(
                {
                    "adj": torch.from_numpy(adj.astype(np.float32)),
                    "target": torch.from_numpy(target.astype(np.float32)),
                    "adj_no_loops": torch.from_numpy(adj_no_loops.astype(np.float32)),
                    "dist": torch.from_numpy(dist.astype(np.int64)),
                    "diameter": int(diameter),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]
