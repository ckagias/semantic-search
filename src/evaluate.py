from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.metrics import mrr_at_k, ndcg_at_k, recall_at_k

DEPTH = 100
N_BOOT = 10_000
BOOT_SEED = 0

METRICS = {
    "recall@10": lambda ranking, rel: recall_at_k(ranking, rel, 10),
    "recall@100": lambda ranking, rel: recall_at_k(ranking, rel, 100),
    "mrr@10": lambda ranking, rel: mrr_at_k(ranking, rel, 10),
    "ndcg@10": lambda ranking, rel: ndcg_at_k(ranking, rel, 10),
}


@dataclass
class Result:
    per_query: pd.DataFrame
    summary: pd.DataFrame


def rank_from_scores(doc_ids, scores, top_n=DEPTH):
    scores = np.asarray(scores, dtype=float)
    order = sorted(range(len(doc_ids)), key=lambda i: (-scores[i], doc_ids[i]))
    return [doc_ids[i] for i in order[:top_n]]


def bootstrap_ci(values, n_boot=N_BOOT, alpha=0.05, seed=BOOT_SEED):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def evaluate(system, queries, qrels, qids, n_boot=N_BOOT, seed=BOOT_SEED):
    rows = {}
    for qid in qids:
        relevant = {doc for doc, grade in qrels[qid].items() if grade > 0}
        ranking = system(queries[qid])[:DEPTH]
        rows[qid] = {name: metric(ranking, relevant) for name, metric in METRICS.items()}
    per_query = pd.DataFrame.from_dict(rows, orient="index")

    summary = pd.DataFrame(
        {name: bootstrap_ci(per_query[name], n_boot, seed=seed) for name in METRICS},
        index=["mean", "lo", "hi"],
    ).T
    return Result(per_query, summary)
