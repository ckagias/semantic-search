import sys

import pandas as pd
from sentence_transformers import SentenceTransformer

from src.baselines import random_system
from src.data import DATA_DIR, load_corpus, load_qrels, load_queries
from src.dense import DenseSearch, encode_corpus
from src.evaluate import METRICS, bootstrap_ci, evaluate
from src.hybrid import Hybrid
from src.keyword import build_bm25
from src.score_dense import BGE_PREFIX, DEFAULT_MODEL
from src.splits import load_splits
from src.train import MODELS_DIR

BGE = "BAAI/bge-base-en-v1.5"
HYBRID = ("wsum", 0.2)
FT_RUNS = {0: 3, 1: 4, 2: 4}
RANDOM_SEEDS = range(5)
OUT = DATA_DIR.parent / "test_per_query.csv"
FT_NAME = "MiniLM fine-tuned, hard negatives"


def ft_run(seed):
    return f"all-MiniLM-L6-v2_hard_s{seed}"


def ft_search(seed, corpus):
    model = SentenceTransformer(str(MODELS_DIR / ft_run(seed)), device="cpu")
    doc_ids, emb, _ = encode_corpus(model, corpus, f"ft_{ft_run(seed)}_epoch{FT_RUNS[seed]}")
    return DenseSearch(model, doc_ids, emb, "")


def check_fine_tuned_models(corpus, queries, train_qrels, val):
    for seed in FT_RUNS:
        log = pd.read_csv(DATA_DIR.parent / f"ft_{ft_run(seed)}_log.csv").set_index("epoch")
        best = log.loc[1:, "ndcg@10"].idxmax()
        assert best == FT_RUNS[seed], f"seed {seed}: best epoch {best}, expected {FT_RUNS[seed]}"
        result = evaluate(ft_search(seed, corpus).system(), queries, train_qrels, val, n_boot=10)
        got, want = result.per_query["ndcg@10"].mean(), log.loc[best, "ndcg@10"]
        assert abs(got - want) < 1e-3, f"seed {seed}: saved model scores {got:.4f} on validation, log says {want:.4f}"
        print(f"seed {seed}: saved model and cached embeddings reproduce validation nDCG@10 {got:.4f} (epoch {best})")


def main():
    if OUT.exists():
        sys.exit(f"{OUT} exists: the test queries have already been scored once. Delete it only to repeat the final scoring on purpose.")

    corpus = load_corpus()
    queries = load_queries()
    train_qrels = load_qrels("train")
    test_qrels = load_qrels("test")
    splits = load_splits()
    test = splits.test
    assert not set(test) & (set(splits.train) | set(splits.val))
    check_fine_tuned_models(corpus, queries, train_qrels, splits.val)

    doc_ids = sorted(corpus)
    bm25 = build_bm25(corpus)
    systems = {f"random, seed {s}": random_system(doc_ids, s) for s in RANDOM_SEEDS}
    systems["stemmed BM25"] = bm25.system()
    models = {}
    for name, model_name, prefix in (("MiniLM pretrained", DEFAULT_MODEL, ""), ("bge-base pretrained", BGE, BGE_PREFIX)):
        model = SentenceTransformer(model_name, device="cpu")
        ids, emb, _ = encode_corpus(model, corpus, model_name)
        models[name] = DenseSearch(model, ids, emb, prefix)
        systems[name] = models[name].system()
    systems["hybrid BM25 + bge-base"] = Hybrid(bm25, models["bge-base pretrained"], doc_ids, *HYBRID).system()
    for seed in FT_RUNS:
        systems[f"{FT_NAME}, seed {seed}"] = ft_search(seed, corpus).system()

    frames, summaries = [], {}
    for name, system in systems.items():
        result = evaluate(system, queries, test_qrels, test)
        frames.append(result.per_query.assign(system=name).rename_axis("qid").reset_index())
        summaries[name] = result.summary
    pd.concat(frames).to_csv(OUT, index=False)
    print(f"\nscored {len(test)} test queries, {len(systems)} systems; per-query scores in {OUT}")

    def row(name):
        s = summaries[name]
        cells = [f"{s.loc['ndcg@10', 'mean']:.3f} ({s.loc['ndcg@10', 'lo']:.3f} to {s.loc['ndcg@10', 'hi']:.3f})"]
        cells += [f"{s.loc[m, 'mean']:.3f}" for m in ("mrr@10", "recall@10", "recall@100")]
        return " | ".join(cells)

    print("\nsystem | nDCG@10 (95% interval) | MRR@10 | recall@10 | recall@100")
    randoms = [summaries[f"random, seed {s}"] for s in RANDOM_SEEDS]
    lo = min(r.loc["ndcg@10", "mean"] for r in randoms)
    hi = max(r.loc["ndcg@10", "mean"] for r in randoms)
    r100 = [r.loc["recall@100", "mean"] for r in randoms]
    print(f"random (5 seeds) | {lo:.3f} to {hi:.3f} | n/a | n/a | {min(r100):.3f} to {max(r100):.3f}")
    for name in ("stemmed BM25", "MiniLM pretrained", "bge-base pretrained", "hybrid BM25 + bge-base"):
        print(f"{name} | {row(name)}")
    for seed in FT_RUNS:
        print(f"{FT_NAME}, seed {seed} | {row(f'{FT_NAME}, seed {seed}')}")

    long = pd.concat(frames)
    wide = long.pivot(index="qid", columns="system", values="ndcg@10")
    wide[FT_NAME] = wide[[f"{FT_NAME}, seed {s}" for s in FT_RUNS]].mean(axis=1)

    def paired(a, b):
        diff = wide[a] - wide[b]
        mean, lo_, hi_ = bootstrap_ci(diff)
        return f"{a} minus {b}: {mean:+.3f} ({lo_:+.3f} to {hi_:+.3f}), better on {(diff > 0).sum()}, worse on {(diff < 0).sum()}"

    print("\npaired nDCG@10 differences on the test queries (fine-tuned = mean of the three seeds per query)")
    for a, b in (
        ("bge-base pretrained", "stemmed BM25"),
        ("hybrid BM25 + bge-base", "stemmed BM25"),
        ("hybrid BM25 + bge-base", "bge-base pretrained"),
        ("MiniLM pretrained", "stemmed BM25"),
        (FT_NAME, "MiniLM pretrained"),
        (FT_NAME, "stemmed BM25"),
    ):
        print(paired(a, b))
    ft_mean = wide[FT_NAME].mean()
    print(f"\nfine-tuned, mean of seeds: nDCG@10 {ft_mean:.3f}")


if __name__ == "__main__":
    main()
