"""Report IX -- measure whether connectivity information survives in the final embeddings
even where the read-out (a single fixed threshold) gets it wrong: "the components are encoded
but not decoded." Three threshold-free/oracle diagnostics on the SAME final embeddings the
model's own read-out uses:

  1. AUROC between within-component and cross-component cosine similarity -- does cosine
     separate the two classes at ALL, regardless of any particular threshold?
  2. Best-possible threshold on cosine (searched directly, not the model's own
     scale/bias-derived one) vs. the model's OWN threshold accuracy on the same pairs -- if the
     oracle threshold does much better, the problem is in the read-out's fixed threshold, not
     in the embeddings.
  3. K-means clustering (K = true number of components, known) on the final embeddings,
     compared against the true component labels via the adjusted Rand index -- if clustering
     recovers the true components even where exact-match/cut fails, the components are encoded
     in the embedding geometry even when the read-out's single global threshold cannot decode
     them.

Works for ANY topology already registered in mechanistic_asym_chains._TOPOLOGY_GENERATORS
(chain, cycle, split_cliques, chorded_cycles, split_regular3) for a 2-way split, OR an
arbitrary K-way split via --sizes (chain or cycle topology, generate_multi_{path,cycle}_split_graph).

    python auroc_cluster_probe.py --checkpoint runs/.../last.pt --topology chain --split 23
    python auroc_cluster_probe.py --checkpoint runs/.../last.pt --sizes 15 15 16 --kway_topology chain
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, adjusted_rand_score
from sklearn.cluster import KMeans

from data import add_self_loops, generate_multi_path_split_graph, generate_multi_cycle_split_graph
from eval_families import load_model
from mechanistic_asym_chains import _TOPOLOGY_GENERATORS, _device


def component_labels_2way(n, a):
    return np.array([0] * a + [1] * (n - a))


def component_labels_kway(sizes):
    labels = []
    for i, s in enumerate(sizes):
        labels += [i] * s
    return np.array(labels)


def get_final_embeddings(model, dev, adj, n_graphs, rng):
    """Returns [G, n, d] final-layer embeddings H, unpermuted back to BASE node coordinates
    (same convention as every heatmap/stagewise probe in this project: average structural
    quantities over many independent relabellings of the same underlying graph)."""
    n = adj.shape[0]
    xs = np.empty((n_graphs, n, n), np.float32)
    invs = []
    for i in range(n_graphs):
        p = rng.permutation(n)
        xs[i] = add_self_loops(adj[np.ix_(p, p)])
        invs.append(np.argsort(p))
    xb = torch.from_numpy(xs).to(dev, torch.float32)
    with torch.no_grad():
        _, h = model.forward_and_embeddings(xb)
    h = h.detach().cpu().numpy()
    out = np.empty_like(h)
    for i, inv in enumerate(invs):
        out[i] = h[i][inv]
    return out


def cosine_matrix(h):
    hn = h / (np.linalg.norm(h, axis=-1, keepdims=True) + 1e-9)
    return hn @ hn.T


def analyze(model, dev, adj, labels, n_graphs, rng, kmeans_seed=0):
    h_all = get_final_embeddings(model, dev, adj, n_graphs, rng)
    n = adj.shape[0]
    off = ~np.eye(n, dtype=bool)
    same = (labels[:, None] == labels[None, :])
    y_true = same[off].astype(int)  # 1 = same component (ground-truth "connected")

    cos_mean = np.zeros((n, n))
    for g in range(h_all.shape[0]):
        cos_mean += cosine_matrix(h_all[g])
    cos_mean /= h_all.shape[0]
    scores = cos_mean[off]

    auroc = roc_auc_score(y_true, scores)

    best_acc, best_thr = 0.0, float(scores.min())
    for t in np.unique(scores):
        acc = ((scores > t).astype(int) == y_true).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, float(t)

    scale = float(model.sim_scale.detach().cpu()) if hasattr(model, "sim_scale") else 1.0
    bias = float(model.sim_bias.detach().cpu()) if hasattr(model, "sim_bias") else 0.0
    model_thr = -bias / scale if scale != 0 else 0.0
    model_acc = float(((scores > model_thr).astype(int) == y_true).mean())

    h_rep = h_all.mean(0)  # [n, d], expected embedding per base node position
    K = len(set(labels.tolist()))
    km = KMeans(n_clusters=K, n_init=10, random_state=kmeans_seed).fit(h_rep)
    ari = float(adjusted_rand_score(labels, km.labels_))

    return {"n": n, "K": K, "auroc": float(auroc),
            "best_threshold": best_thr, "best_acc": float(best_acc),
            "model_threshold": float(model_thr), "model_acc": model_acc,
            "kmeans_ari": ari}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", default=None,
                    help="if given, save results as JSON here; otherwise just print")
    ap.add_argument("--topology",
                    choices=list(_TOPOLOGY_GENERATORS.keys()), default=None,
                    help="2-way split test (mutually exclusive with --sizes)")
    ap.add_argument("--split", type=int, default=None, help="short-component size a (2-way)")
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="K-way split sizes s1 s2 ... sK, summing to n (mutually exclusive "
                         "with --topology/--split)")
    ap.add_argument("--kway_topology", choices=["chain", "cycle"], default="chain",
                    help="only used with --sizes")
    ap.add_argument("--n_graphs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    if (args.topology is not None) == (args.sizes is not None):
        raise ValueError("pass exactly one of --topology/--split (2-way) or --sizes (K-way)")

    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    if readout != "similarity":
        raise NotImplementedError("this probe reads model.sim_scale/sim_bias -- similarity "
                                   "read-out only")
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")

    rng = np.random.default_rng(args.seed)
    if args.sizes is not None:
        if sum(args.sizes) != n:
            raise ValueError(f"--sizes {args.sizes} does not sum to n={n}")
        build = generate_multi_cycle_split_graph if args.kway_topology == "cycle" \
            else generate_multi_path_split_graph
        adj = build(n, tuple(args.sizes))
        labels = component_labels_kway(args.sizes)
        tag = f"sizes_{'_'.join(map(str, args.sizes))}_{args.kway_topology}"
    else:
        if args.split is None:
            raise ValueError("--split is required with --topology")
        adj = _TOPOLOGY_GENERATORS[args.topology](n, args.split)
        labels = component_labels_2way(n, args.split)
        tag = f"{args.topology}_a{args.split}"

    result = analyze(model, dev, adj, labels, args.n_graphs, rng)
    result["tag"] = tag
    print(json.dumps(result, indent=2))
    print(f"\n=> AUROC={result['auroc']:.3f} | model read-out acc={result['model_acc']:.3f} "
          f"vs. best-possible-threshold acc={result['best_acc']:.3f} "
          f"| K-means ARI={result['kmeans_ari']:.3f} (K={result['K']})")
    if result["best_acc"] - result["model_acc"] > 0.05 and result["kmeans_ari"] > 0.8:
        print("=> components are encoded (AUROC high, clustering recovers them) but the "
              "model's own fixed threshold does not decode them well: ENCODED BUT NOT DECODED.")

    if args.output_dir:
        out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
        (out / f"auroc_cluster_{tag}.json").write_text(json.dumps(result, indent=2))
        print(f"saved -> {out}/auroc_cluster_{tag}.json")


if __name__ == "__main__":
    main()
