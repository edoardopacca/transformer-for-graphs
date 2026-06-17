"""Reference connectivity-matrix predictors (Report V).

To turn ``the model looks like it does DFS'' into a measurement, we compare the
transformer's error pattern against two explicit algorithms that compute a
connectivity matrix in opposite ways:

  matrix_power_connectivity(adj, L)
      The DISTANCE-bounded predictor: 1[(A+I)^{3^L} > 0]. Two nodes are called
      connected iff a walk of length <= 3^L joins them (with self-loops, a walk of
      length <= 3^L exists iff their shortest-path distance is <= 3^L). This is what
      an L-layer model doing matrix powering computes; it is FLAT in clique size and
      bounded only by shortest-path distance.

  bounded_dfs_connectivity(adj, budget)
      The VISIT-bounded predictor: from each start node run a depth-first search
      that stops after visiting ``budget`` nodes, and mark everything it reached as
      connected to that start. Bounded by the NUMBER OF NODES traversed, not by
      distance -- so it gets a large, dense component wrong (it runs out of budget
      before reaching the far side) even when that side is only a few hops away.

These make OPPOSITE predictions on two dense cliques joined by one bridge edge: the
matrix-power oracle connects them (the bridge is <= 3 hops); the bounded-DFS oracle
fails once a clique is larger than the budget (it exhausts the budget inside the near
clique). Which oracle the transformer's mistakes resemble -- especially on the
entries where the two oracles DISAGREE -- is the test of whether it learned matrix
powering or a bounded DFS.

Pure NumPy, no GPU; fast enough to run inline in the eval scripts.
"""
from __future__ import annotations

import numpy as np


def reach_within_hops_connectivity(adj_no_loops: np.ndarray, hops: int) -> np.ndarray:
    """1[(A+I)^hops > 0]: connected within ``hops`` message-passing steps. The
    generic distance-bounded predictor; matrix_power_connectivity is hops = 3**L."""
    n = adj_no_loops.shape[0]
    m = (adj_no_loops > 0).astype(np.float64) + np.eye(n)
    reach = np.linalg.matrix_power(m, int(hops))
    return (reach > 0).astype(np.int8)


def matrix_power_connectivity(adj_no_loops: np.ndarray, n_layers: int) -> np.ndarray:
    """The matrix-powering predictor for an ``n_layers``-layer model: connected iff
    shortest-path distance <= 3^L (capacity 9 for L=2)."""
    return reach_within_hops_connectivity(adj_no_loops, 3 ** int(n_layers))


def bounded_bfs_connectivity(adj_no_loops: np.ndarray, budget: int,
                             symmetric: bool = True) -> np.ndarray:
    """Connectivity matrix of a breadth-first traversal that visits at most
    ``budget`` nodes per start (the ``budget`` nodes NEAREST to the start), and
    marks them connected to it. This is the cleanest ``gets stuck'' reference for the
    bridged-cliques probe: from a node in one clique, BFS first fills its own clique
    (all c-1 neighbours at distance 1) and only crosses the bridge once the budget
    exceeds the near-clique size -- so it fails the cross pairs for budget <= c,
    sharply, however few hops away the far clique is. Bounded by NODES visited, not
    distance: the opposite of matrix_power_connectivity. ``symmetric`` as below."""
    n = adj_no_loops.shape[0]
    budget = int(budget)
    neigh = [np.where(adj_no_loops[u] > 0)[0].tolist() for u in range(n)]
    R = np.zeros((n, n), np.int8)
    for s in range(n):
        seen = np.zeros(n, bool)
        seen[s] = True
        q = [s]
        head = 0
        n_visited = 0
        while head < len(q) and n_visited < budget:
            u = q[head]; head += 1
            R[s, u] = 1
            n_visited += 1
            for v in neigh[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        R[s, s] = 1
    if symmetric:
        R = (R | R.T).astype(np.int8)
    return R


def bounded_dfs_connectivity(adj_no_loops: np.ndarray, budget: int,
                             symmetric: bool = True) -> np.ndarray:
    """Connectivity matrix of a DFS that visits at most ``budget`` nodes per start.

    From each start node ``s`` we run an iterative depth-first search (neighbours in
    ascending index order) and stop after popping ``budget`` nodes; everything popped
    is marked connected to ``s``. The raw matrix R is generally ASYMMETRIC: s may
    reach v within budget while v does not reach s within budget (the DFS signature).

    ``symmetric``: if True (default) return R | R.T -- a pair is called connected if
    the DFS reached it from EITHER end (the most generous symmetric reading, to be
    compared against the model's possibly-asymmetric output via majority logic). If
    False, return the raw asymmetric R.
    """
    n = adj_no_loops.shape[0]
    budget = int(budget)
    neigh = [np.where(adj_no_loops[u] > 0)[0].tolist() for u in range(n)]
    R = np.zeros((n, n), np.int8)
    for s in range(n):
        seen = np.zeros(n, bool)
        seen[s] = True
        # push neighbours in reverse so the smallest index is popped (visited) first
        stack = [s]
        n_visited = 0
        while stack and n_visited < budget:
            u = stack.pop()
            R[s, u] = 1
            n_visited += 1
            for v in sorted(neigh[u], reverse=True):
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        R[s, s] = 1
    if symmetric:
        R = (R | R.T).astype(np.int8)
    return R
