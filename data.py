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
    n = adj_no_loops.shape[0]
    d = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        d[i] = _bfs_distances(adj_no_loops, i)
    return d


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
