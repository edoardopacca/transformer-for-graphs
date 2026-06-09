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


def generate_two_cliques_graph(n: int, k: int) -> np.ndarray:
    if n != 2 * k:
        raise ValueError(f"TwoCliques requires n == 2*k, got n={n}, k={k}")
    adj = np.zeros((n, n), dtype=np.float32)
    for start in (0, k):
        idx = np.arange(start, start + k)
        adj[np.ix_(idx, idx)] = 1.0
        np.fill_diagonal(adj[start : start + k, start : start + k], 0.0)
    return adj


def generate_one_chain_graph(n: int) -> np.ndarray:
    """Single path (chain) over all n nodes: 0-1-2-...-(n-1). One connected
    component. Used as the negative ("1 chain") class in the components task."""
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj


def generate_path_union_graph(n: int, rng: np.random.Generator,
                              max_paths: int = 4) -> np.ndarray:
    """Disjoint union of 1..max_paths paths that partition all n nodes.

    Designed to exercise long-range reachability: with k=1 the graph is a single
    path of n nodes (shortest-path distances up to n-1), while k>1 introduces
    genuine cross-component disconnections so the model cannot trivially predict
    "all connected". Node order is NOT permuted here (do it at the call site)."""
    k = int(rng.integers(1, max_paths + 1))
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
