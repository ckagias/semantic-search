import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import DATA_DIR
from src.evaluate import bootstrap_ci

DATA = DATA_DIR.parent
OUT = DATA_DIR.parent.parent / "docs" / "img"

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"
COLORS = {
    "bge-base": BLUE,
    "BM25": ORANGE,
    "MiniLM pretrained": AQUA,
    "MiniLM fine-tuned": VIOLET,
    "hybrid": MAGENTA,
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "figure.dpi": 160,
    }
)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", OUT / name)


def test_results():
    df = pd.read_csv(DATA / "test_per_query.csv")
    rows = [
        ("hybrid BM25 + bge-base", "hybrid"),
        ("bge-base pretrained", "bge-base"),
        ("stemmed BM25", "BM25"),
        ("MiniLM fine-tuned, hard negatives, seed 0", "MiniLM fine-tuned"),
        ("MiniLM fine-tuned, hard negatives, seed 1", "MiniLM fine-tuned"),
        ("MiniLM fine-tuned, hard negatives, seed 2", "MiniLM fine-tuned"),
        ("MiniLM pretrained", "MiniLM pretrained"),
    ]
    labels = ["hybrid BM25 + bge-base", "bge-base", "BM25 (stemmed)", "MiniLM fine-tuned, seed 0", "MiniLM fine-tuned, seed 1", "MiniLM fine-tuned, seed 2", "MiniLM pretrained"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, ((system, key), label) in enumerate(zip(rows, labels)):
        y = len(rows) - 1 - i
        mean, lo, hi = bootstrap_ci(df[df["system"] == system]["ndcg@10"])
        ax.plot([lo, hi], [y, y], color=COLORS[key], linewidth=2, solid_capstyle="round")
        ax.plot(mean, y, "o", color=COLORS[key], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(hi + 0.006, y, f"{mean:.3f}", va="center", ha="left", color=INK, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlim(0.58, 0.83)
    ax.set_xlabel("nDCG@10, 300 test queries (95% bootstrap interval over queries)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Test results, each system scored once")
    save(fig, "test_results.png")


def overlap_terciles():
    df = pd.read_csv(DATA / "error_analysis_val.csv")
    df["bucket"] = pd.qcut(df["overlap"], 3, labels=["low", "mid", "high"])
    names = [("bm25_rr", "BM25", ORANGE), ("pretrained_rr", "MiniLM pretrained", AQUA), ("tuned_rr", "MiniLM fine-tuned", VIOLET)]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    width = 0.26
    for j, (col, label, color) in enumerate(names):
        values = [df[df["bucket"] == b][col].mean() for b in ("low", "mid", "high")]
        x = np.arange(3) + (j - 1) * (width + 0.02)
        ax.bar(x, values, width, color=color, label=label, edgecolor=SURFACE, linewidth=2)
        for xi, v in zip(x, values):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    sizes = df["bucket"].value_counts()
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{b} overlap\n({sizes[b]} queries)" for b in ("low", "mid", "high")])
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("mean reciprocal rank, validation")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", ncols=3, fontsize=9)
    ax.set_title("The fine-tuned model wins only when the query shares few words with its answer", fontsize=10)
    save(fig, "overlap_terciles.png")


def ablation_seeds():
    cats = [("pretrained", None), ("100 queries", "hard_n100"), ("200 queries", "hard_n200"), ("415 queries", "hard")]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.axhline(0.698, color=ORANGE, linewidth=2, linestyle=(0, (5, 3)))
    ax.text(3.45, 0.7, "BM25 0.698", color=INK, va="bottom", ha="right", fontsize=9)
    means = []
    for i, (label, key) in enumerate(cats):
        if key is None:
            log = pd.read_csv(DATA / "ft_all-MiniLM-L6-v2_hard_s0_log.csv")
            values = [float(log.loc[log["epoch"] == 0, "ndcg@10"].iloc[0])]
        else:
            values = []
            for seed in range(3):
                log = pd.read_csv(DATA / f"ft_all-MiniLM-L6-v2_{key}_s{seed}_log.csv")
                values.append(float(log[log["epoch"] > 0]["ndcg@10"].max()))
        offsets = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else [0]
        for off, v in zip(offsets, values):
            ax.plot(i + off, v, "o", color=VIOLET, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        mean = float(np.mean(values))
        means.append((i, mean))
        ax.text(i, max(values) + 0.004, f"{mean:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.plot(*zip(*means), color=VIOLET, linewidth=1.5, alpha=0.6)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c[0] for c in cats])
    ax.set_xlim(-0.4, 3.5)
    ax.set_ylim(0.61, 0.72)
    ax.set_ylabel("best-epoch nDCG@10, validation")
    ax.grid(axis="x", visible=False)
    ax.set_title("More training data helps; the seeds agree once there is enough")
    ax.text(0.01, -0.2, "Fine-tuned MiniLM with hard negatives; dots are seeds 0, 1 and 2, the line joins their means.", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    save(fig, "ablation_seeds.png")


def ceiling():
    df = pd.read_csv(DATA / "ceiling_val_per_query.csv")
    df = df[df["k"] != "up to 100"].copy()
    df["k"] = df["k"].astype(int)
    series = [("bge-base", "bge-base", BLUE), ("hybrid BM25 + bge-base", "hybrid", MAGENTA), ("stemmed BM25", "BM25", ORANGE), ("MiniLM pretrained", "MiniLM pretrained", AQUA), ("MiniLM fine-tuned", "MiniLM fine-tuned", VIOLET)]
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    end_y = {}
    for system, label, color in series:
        part = df[df["system"] == system].groupby("k")["oracle_ndcg@10"].mean()
        ax.plot(part.index, part.values, "-o", color=color, linewidth=2, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        end_y[label] = part.values[-1]
    nudge = {"hybrid": -0.0, "bge-base": 0.008, "BM25": -0.004, "MiniLM pretrained": 0.006, "MiniLM fine-tuned": -0.002}
    for label, y in end_y.items():
        ax.text(103, y + nudge[label], label, va="center", ha="left", fontsize=9, color=INK)
    ax.set_xticks([10, 20, 50, 100])
    ax.set_xlim(8, 135)
    ax.set_ylim(0.74, 1.0)
    ax.set_xlabel("candidates kept from the first stage")
    ax.set_ylabel("nDCG@10 of a perfect re-ranker")
    ax.set_title("Ceiling for a re-ranker, validation")
    ax.text(0.01, -0.27, "The hybrid's own nDCG@10 is 0.794 and bge-base's 0.759: the gap to the line is the headroom left to a re-ranker.", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    save(fig, "ceiling.png")


def overlap_probe():
    df = pd.read_csv(DATA / "probe_overlap_val_per_query.csv")
    systems = [("bge-base", "bge-base"), ("MiniLM fine-tuned", "MiniLM fine-tuned"), ("MiniLM pretrained", "MiniLM pretrained"), ("BM25", "BM25 (stemmed)")]
    conds = [("control_ndcg", "no change", INK), ("random_ndcg", "random informative words deleted", AQUA), ("overlap_deleted_ndcg", "overlapping words deleted", ORANGE)]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, (name, label) in enumerate(systems):
        y = len(systems) - 1 - i
        part = df[df["system"] == name]
        vals = [part[c].mean() for c, _, _ in conds]
        ax.plot([min(vals), max(vals)], [y, y], color=GRID, linewidth=3, zorder=1)
        for (c, _, color), v in zip(conds, vals):
            ax.plot(v, y, "o", color=color, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.text(vals[0] + 0.02, y + 0.22, f"{vals[0]:.2f}", fontsize=8.5, ha="center", color=INK)
        ax.text(vals[2] - 0.02, y + 0.22, f"{vals[2]:.2f}", fontsize=8.5, ha="center", color=INK)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels([s[1] for s in systems][::-1])
    ax.set_xlim(-0.03, 0.88)
    ax.set_ylim(-0.6, len(systems) - 0.2)
    ax.set_xlabel("nDCG@10 on 156 validation queries")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=8, markeredgecolor=SURFACE, label=l) for _, l, c in conds]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncols=3, fontsize=8.5)
    ax.set_title("Every system depends on the words it shares with the answer")
    save(fig, "overlap_probe.png")


def search_scale():
    scale = pd.read_csv(DATA / "bench_scale.csv")
    ann = pd.read_csv(DATA / "bench_ann.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1, 1.15]})
    for model, color, label in (("MiniLM (384)", AQUA, "MiniLM, 384 numbers"), ("bge-base (768)", BLUE, "bge-base, 768 numbers")):
        part = scale[scale["model"] == model]
        a.plot(part["documents"], part["matvec + partition median ms"], "-o", color=color, linewidth=2, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        x, y, ha = (5.6e3, 1.3, "left") if model.startswith("bge") else (4.8e5, 3.5, "right")
        a.text(x, y, label, ha=ha, fontsize=9, color=INK)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("vectors in the index")
    a.set_ylabel("exact search, ms per query")
    a.set_title("Exact search grows with the bytes read")
    big = ann[ann["corpus"] == "synthetic"]
    flat = big[big["index"].str.startswith("flat")]
    for prefix, color, label in (("IVF", ORANGE, "IVF"), ("HNSW", VIOLET, "HNSW")):
        part = big[big["index"].str.startswith(prefix)]
        b.plot(part["median ms"], part["source-document overlap@10"], "-o", color=color, linewidth=2, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        x, y = (0.9, 0.58) if prefix == "IVF" else (0.14, 0.8)
        b.text(x, y, label, fontsize=9, color=INK)
    b.plot(flat["median ms"], flat["source-document overlap@10"], "s", color=INK, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
    b.text(flat["median ms"].iloc[0], 0.955, "exact\n(flat)", ha="center", va="top", fontsize=9, color=INK)
    b.set_xscale("log")
    b.set_xlabel("ms per query, 518,300 vectors")
    b.set_ylabel("source documents found\n(share of exact top 10)")
    b.set_ylim(0.25, 1.03)
    b.set_xlim(0.05, 120)
    b.set_title("Approximate search: speed against accuracy")
    fig.text(0.01, -0.03, "Left: 5,183 SciFact vectors tiled up to 518,300. Right: synthetic corpus of noisy copies; the timings transfer, the accuracy figures do not.", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    save(fig, "search_scale.png")


def main():
    for f in (test_results, overlap_terciles, ablation_seeds, ceiling, overlap_probe, search_scale):
        f()


if __name__ == "__main__":
    main()
