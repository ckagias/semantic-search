import re

import pandas as pd

from src.data import DATA_DIR

LOG = re.compile(r"ft_(?P<model>.+)_(?P<negatives>in-batch|hard)(?:_n(?P<n>\d+))?_s(?P<seed>\d+)_log\.csv")


def collect(model="all-MiniLM-L6-v2"):
    rows = []
    for path in sorted(DATA_DIR.parent.glob("ft_*_log.csv")):
        m = LOG.fullmatch(path.name)
        if not m or m["model"] != model:
            continue
        log = pd.read_csv(path)
        best = log[log["epoch"] > 0].sort_values("ndcg@10", ascending=False).iloc[0]
        rows.append(
            {
                "negatives": m["negatives"],
                "queries": int(m["n"]) if m["n"] else "all",
                "seed": int(m["seed"]),
                "pretrained": log.loc[log["epoch"] == 0, "ndcg@10"].item(),
                "best_epoch": int(best["epoch"]),
                "ndcg@10": best["ndcg@10"],
                "recall@10": best["recall@10"],
                "recall@100": best["recall@100"],
            }
        )
    return pd.DataFrame(rows)


def summarize(runs):
    grouped = runs.groupby(["negatives", "queries"], sort=False)
    out = grouped.agg(
        seeds=("seed", "count"),
        mean=("ndcg@10", "mean"),
        low=("ndcg@10", "min"),
        high=("ndcg@10", "max"),
        recall10=("recall@10", "mean"),
        recall100=("recall@100", "mean"),
    )
    return out.reset_index()


if __name__ == "__main__":
    runs = collect()
    print(runs.round(3).to_string(index=False))
    print()
    print(summarize(runs).round(3).to_string(index=False))
