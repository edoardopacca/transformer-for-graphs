"""Report VII -- render the weight- and attention-heatmaps (local, no GPU).

Reads runs/report7/heatmaps/<tag>/{heatmap_data.npz,raw_weights.npz} (from
mechanistic_heatmaps.py) and runs/report7/mechanistic/<tag>/weights_summary.json
(from mechanistic_asym_chains.py, Tier 1), pools across the four seeds where the
quantity is split-independent (the static weight matrices) or reports one
representative seed for the per-graph attention heatmaps (attention patterns are
consistent across seeds, as Tier 1's leak-fraction figure already showed --
spread <1 point -- so one seed is representative; not averaged across seeds
because averaging attention PATTERNS across seeds with different learned bases
would blur genuine structure, unlike averaging a scalar leak fraction).

    python plot_mechanistic_heatmaps.py --seed 1000
"""
import argparse, json
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HEAT_ROOT = Path("runs/report7/heatmaps")
MECH_ROOT = Path("runs/report7/mechanistic")
OUT = Path("runs/report7/report7_figs")
FIG_PREFIX = "r7"   # reset from --report_root in main() (e.g. "r8" for report8)
PAIR = (4, 20)   # solved vs failed, the recurring comparison pair in this report


def heat(ax, mat, title, cmap="RdBu_r", vlim=None, cbar=True, diverging=True):
    if diverging:
        v = vlim if vlim is not None else np.abs(mat).max()
        im = ax.imshow(mat, cmap=cmap, vmin=-v, vmax=v, aspect="auto")
    else:
        vmax = vlim if vlim is not None else mat.max()
        im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=9)
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def fig_weight_geometry(seed_tag, suffix=""):
    ws = json.load((MECH_ROOT / seed_tag / "weights_summary.json").open())
    raw = np.load(HEAT_ROOT / seed_tag / "raw_weights.npz")
    if ws.get("readout_kind") == "similarity":
        # no W_out / per-target row for this read-out -- only E_in has a
        # geometry to check; report the two read-out scalars in the title.
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        heat(axes[0], raw["W_in"].T, r"$E_{\mathrm{in}}=W_{\mathrm{in}}^\top$ (neighbour-labels $\times$ $d_{\mathrm{model}}$)", cmap="RdBu_r")
        heat(axes[1], np.array(ws["cos_in"]), r"$\cos(e_k,e_l)$", cmap="RdBu_r", vlim=1.0)
        axes[2].plot(ws["norms_in"], label=r"$\|e_k\|$")
        axes[2].set_xlabel("label index $k$"); axes[2].legend(fontsize=8)
        axes[2].set_title("read-in row norms", fontsize=9); axes[2].grid(alpha=0.3)
        fig.suptitle(f"Read-in ($W_{{\\mathrm{{in}}}}$) geometry -- {seed_tag} (similarity read-out: "
                     f"scale={ws['sim_scale']:.2f}, bias={ws['sim_bias']:.2f}, no $W_{{\\mathrm{{out}}}}$)")
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        heat(axes[0, 0], raw["W_out"], r"$W_{\mathrm{out}}$ (targets $\times$ $d_{\mathrm{model}}$)", cmap="RdBu_r")
        heat(axes[0, 1], raw["W_in"].T, r"$E_{\mathrm{in}}=W_{\mathrm{in}}^\top$ (neighbour-labels $\times$ $d_{\mathrm{model}}$)", cmap="RdBu_r")
        heat(axes[0, 2], np.array(ws["alignment_ein_wout"]), r"$E_{\mathrm{in}}W_{\mathrm{out}}^\top$ (skip-connection alignment)", cmap="RdBu_r")
        heat(axes[1, 0], np.array(ws["cos_out"]), r"$\cos(w_j,w_k)$", cmap="RdBu_r", vlim=1.0)
        heat(axes[1, 1], np.array(ws["cos_in"]), r"$\cos(e_k,e_l)$", cmap="RdBu_r", vlim=1.0)
        axes[1, 2].plot(ws["norms_out"], label=r"$\|w_j\|$")
        axes[1, 2].plot(ws["norms_in"], label=r"$\|e_k\|$")
        axes[1, 2].set_xlabel("target / label index $j$"); axes[1, 2].legend(fontsize=8)
        axes[1, 2].set_title("read-out / read-in row norms", fontsize=9)
        axes[1, 2].grid(alpha=0.3)
        fig.suptitle(f"Read-out ($W_{{\\mathrm{{out}}}}$) and read-in ($W_{{\\mathrm{{in}}}}$) geometry -- {seed_tag}")
    fig.tight_layout()
    p = OUT / f"{FIG_PREFIX}_heatmap_weight_geometry{suffix}.png"
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def fig_qkvo_raw_weights(seed_tag, suffix=""):
    raw = np.load(HEAT_ROOT / seed_tag / "raw_weights.npz")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    names = [("WQ0", r"$W_Q$ layer 0"), ("WK0", r"$W_K$ layer 0"), ("WV0", r"$W_V$ layer 0"), ("WO0", r"$W_O$ layer 0"),
             ("WQ1", r"$W_Q$ layer 1"), ("WK1", r"$W_K$ layer 1"), ("WV1", r"$W_V$ layer 1"), ("WO1", r"$W_O$ layer 1")]
    for ax, (key, title) in zip(axes.flat, names):
        heat(ax, raw[key], title, cmap="RdBu_r")
    fig.suptitle(f"Raw attention projection weight matrices ($d_{{\\mathrm{{model}}}}\\times d_{{\\mathrm{{model}}}}$) -- {seed_tag}")
    fig.tight_layout()
    p = OUT / f"{FIG_PREFIX}_heatmap_qkvo_weights{suffix}.png"
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)

    fig, ax = plt.subplots(figsize=(7, 5))
    extra = [("W_in", r"$W_{\mathrm{in}}$")]
    if "sv_W_out" in raw.files:
        extra.append(("W_out", r"$W_{\mathrm{out}}$"))  # absent for the similarity read-out
    for key, title in names + extra:
        sv = raw[f"sv_{key}"]
        ax.plot(sv / sv[0], label=title, alpha=0.8)
    ax.set_yscale("log"); ax.set_xlabel("singular value index"); ax.set_ylabel("relative singular value")
    ax.set_title(f"Singular value spectra (normalised to $\\sigma_1$) -- {seed_tag}")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = OUT / f"{FIG_PREFIX}_heatmap_singular_values{suffix}.png"
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def fig_attention_scores_alpha(seed_tag, pair=PAIR, suffix=""):
    d = np.load(HEAT_ROOT / seed_tag / "heatmap_data.npz")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for col, a in enumerate(pair):
        for li in (0, 1):
            ax = axes[li, col * 2]
            heat(ax, d[f"a{a}__scores{li}"], f"scores layer {li}, a={a}", cmap="RdBu_r")
            ax = axes[li, col * 2 + 1]
            heat(ax, d[f"a{a}__alpha{li}"], rf"$\alpha$ layer {li}, a={a}", cmap="viridis", diverging=False)
    fig.suptitle(f"Attention scores $S=QK^\\top/\\sqrt{{d_h}}$ (left of each pair) and normalized-ReLU "
                f"$\\alpha$ (right) -- {seed_tag}, node order = base graph position (short then long)")
    fig.tight_layout()
    p = OUT / f"{FIG_PREFIX}_heatmap_attention_scores{suffix}.png"
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def fig_rollout_contrib(seed_tag, pair=PAIR, suffix=""):
    d = np.load(HEAT_ROOT / seed_tag / "heatmap_data.npz")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for col, a in enumerate(pair):
        heat(axes[0, col], d[f"a{a}__contrib_exact"],
             f"real contribution $\\|\\partial h_i^{{(2)}}/\\partial h_k^{{(0)}}\\|_F$, a={a}",
             cmap="viridis", diverging=False)
        heat(axes[1, col], d[f"a{a}__contrib1"], f"layer-1 message contribution $\\alpha_{{ij}}\\|W_O v_j\\|$, a={a}", cmap="viridis", diverging=False)
    for row, a in enumerate(pair):
        axes[row, 2].plot(d[f"a{a}__row_mass0"], label="layer 0", color="#1b9e77")
        axes[row, 2].plot(d[f"a{a}__row_mass1"], label="layer 1", color="#d95f02")
        axes[row, 2].axvline(len(d[f"a{a}__short_idx"]) - 0.5, color="gray", ls=":", label="component boundary")
        axes[row, 2].set_title(f"row-mass $\\sum_j\\alpha_{{ij}}$ vs node position, a={a}", fontsize=9)
        axes[row, 2].set_xlabel("base node position (short path first, then long)")
        axes[row, 2].legend(fontsize=7); axes[row, 2].grid(alpha=0.3)
    fig.suptitle(f"Real node-to-node contribution, message contribution, and row-mass -- {seed_tag}")
    fig.tight_layout()
    p = OUT / f"{FIG_PREFIX}_heatmap_rollout_contrib{suffix}.png"
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def fig_contrib_matrix_only(seed_tag, pair=PAIR, suffix=""):
    """Just the raw C_ik = ||dh_i^(2)/dh_k^(0)||_F matrix, one panel per split,
    with no message-contribution or row-mass panels alongside it -- meant to sit
    directly under the aggregate leak-fraction figure (fig:r7attn) in
    S:res-attention, so the reader can see the matrix that curve is summarising
    before reaching the fuller multi-panel figure in S:res-heatmaps."""
    d = np.load(HEAT_ROOT / seed_tag / "heatmap_data.npz")
    fig, axes = plt.subplots(1, len(pair), figsize=(6.5 * len(pair), 5.5))
    if len(pair) == 1:
        axes = [axes]
    for ax, a in zip(axes, pair):
        mat = d[f"a{a}__contrib_exact"]
        heat(ax, mat, f"$C_{{ik}}=\\|\\partial h_i^{{(2)}}/\\partial h_k^{{(0)}}\\|_F$, a={a}",
             cmap="viridis", diverging=False)
        ax.axvline(len(d[f"a{a}__short_idx"]) - 0.5, color="white", ls=":", lw=1)
        ax.axhline(len(d[f"a{a}__short_idx"]) - 0.5, color="white", ls=":", lw=1)
        ax.set_xlabel("source node $k$ (base position, short then long)")
        ax.set_ylabel("query node $i$ (base position, short then long)")
    fig.suptitle(f"Exact node-to-node contribution matrix -- {seed_tag} (dotted lines: component boundary)")
    fig.tight_layout()
    p = OUT / f"{FIG_PREFIX}_heatmap_contrib_matrix{suffix}.png"
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def fig_qkv_nodes(seed_tag, pair=PAIR, layer=1, suffix=""):
    d = np.load(HEAT_ROOT / seed_tag / "heatmap_data.npz")
    # per-node Q/K/V heatmaps [n, head_dim] for one representative split
    a = pair[1]  # the failing (balanced) split -- the more informative one to inspect
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, title in zip(axes2, ("q", "k", "v"), ("Q", "K", "V")):
        heat(ax, d[f"a{a}__{key}{layer}"], f"{title} layer {layer}, a={a} (node $\\times$ head-dim)", cmap="RdBu_r")
        ax.set_xlabel("head dim"); ax.set_ylabel("base node position")
    fig2.suptitle(f"Per-node Q/K/V, layer {layer} -- {seed_tag}")
    fig2.tight_layout()
    p2 = OUT / f"{FIG_PREFIX}_heatmap_qkv_nodes{suffix}.png"
    fig2.savefig(p2, dpi=150); plt.close(fig2); print("saved", p2)

    # norms along chain position, both splits, both layers
    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, a in zip(axes3, pair):
        for li, ls in zip((0, 1), ("-", "--")):
            for key, col in zip(("q", "k", "v"), ("#1b9e77", "#d95f02", "#7570b3")):
                norms = np.linalg.norm(d[f"a{a}__{key}{li}"], axis=-1)
                ax.plot(norms, ls, color=col, alpha=0.9 if li == 1 else 0.5,
                        label=f"{key} layer {li}" if a == pair[0] else None)
        ax.axvline(len(d[f"a{a}__short_idx"]) - 0.5, color="gray", ls=":")
        ax.set_title(f"a={a}", fontsize=10); ax.set_xlabel("base node position"); ax.grid(alpha=0.3)
    axes3[0].set_ylabel(r"$\|q_i\|,\|k_i\|,\|v_i\|$"); axes3[0].legend(fontsize=7, ncol=2)
    fig3.suptitle(f"Per-node Q/K/V norms along the chain (dashed=layer 1, solid=layer 0) -- {seed_tag}")
    fig3.tight_layout()
    p3 = OUT / f"{FIG_PREFIX}_heatmap_qkv_norms{suffix}.png"
    fig3.savefig(p3, dpi=150); plt.close(fig3); print("saved", p3)

    # cosine similarity matrices for Q/K/V, one representative split/layer
    def cosmat(M):
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        return Mn @ Mn.T
    fig4, axes4 = plt.subplots(2, 3, figsize=(15, 9))
    for col, a in enumerate(pair):
        for i, key in enumerate(("q", "k", "v")):
            heat(axes4[col, i], cosmat(d[f"a{a}__{key}{layer}"]), f"cos({key},{key}) layer {layer}, a={a}",
                cmap="RdBu_r", vlim=1.0)
    fig4.suptitle(f"Per-node Q/K/V cosine similarity, layer {layer} -- {seed_tag}")
    fig4.tight_layout()
    p4 = OUT / f"{FIG_PREFIX}_heatmap_qkv_cosine{suffix}.png"
    fig4.savefig(p4, dpi=150); plt.close(fig4); print("saved", p4)


def main():
    global HEAT_ROOT, MECH_ROOT, OUT, FIG_PREFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag_prefix", default="n40_pathunion",
                    help="checkpoint family tag, e.g. n40_pathunion, n64_pathunion, n64_er")
    ap.add_argument("--seed", type=int, default=1000, help="representative seed for the per-graph attention figures")
    ap.add_argument("--pair", type=int, nargs=2, default=list(PAIR), help="solved vs failed split to compare")
    ap.add_argument("--suffix", default="", help="appended to output figure filenames (default: derived from tag_prefix)")
    ap.add_argument("--report_root", default="report7",
                    help="runs/<report_root>/{heatmaps,mechanistic,<report_root>_figs} -- "
                         "e.g. report8 for the two-cycles falsification test")
    args = ap.parse_args()
    HEAT_ROOT = Path(f"runs/{args.report_root}/heatmaps")
    MECH_ROOT = Path(f"runs/{args.report_root}/mechanistic")
    OUT = Path(f"runs/{args.report_root}/{args.report_root}_figs")
    FIG_PREFIX = args.report_root.replace("report", "r")
    pair = tuple(args.pair)
    suffix = args.suffix if args.suffix else (f"_{args.tag_prefix}" if args.tag_prefix != "n40_pathunion" else "")
    OUT.mkdir(parents=True, exist_ok=True)
    seed_tag = f"{args.tag_prefix}_seed{args.seed}"
    if not (HEAT_ROOT / seed_tag).exists():
        print(f"no heatmap data under {HEAT_ROOT / seed_tag} -- run mechanistic_heatmaps.py first"); return
    fig_weight_geometry(seed_tag, suffix)
    fig_qkvo_raw_weights(seed_tag, suffix)
    fig_attention_scores_alpha(seed_tag, pair, suffix)
    fig_rollout_contrib(seed_tag, pair, suffix)
    fig_contrib_matrix_only(seed_tag, pair, suffix)
    fig_qkv_nodes(seed_tag, pair, suffix=suffix)


if __name__ == "__main__":
    main()
