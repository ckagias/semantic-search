import os
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.data import DATA_DIR, load_corpus, load_queries
from src.dense import encode_corpus
from src.evaluate import DEPTH
from src.score_dense import BGE_PREFIX, DEFAULT_MODEL
from src.splits import load_splits

BGE = "BAAI/bge-base-en-v1.5"
COPIES = [1, 10, 100]
REPEATS = 3
OUT = DATA_DIR.parent / "bench_scale.csv"


def top_sorted(scores, k=DEPTH):
    return np.argsort(-scores, kind="stable")[:k]


def top_partition(scores, k=DEPTH):
    part = np.argpartition(-scores, k)[:k]
    return part[np.argsort(-scores[part], kind="stable")]


def per_query_ms(fn, emb, queries):
    fn(emb @ queries[0])
    times = []
    for _ in range(REPEATS):
        for q in queries:
            t = time.perf_counter()
            fn(emb @ q)
            times.append(time.perf_counter() - t)
    return 1000 * np.array(times)


def matvec_ms(emb, queries):
    emb @ queries[0]
    times = []
    for _ in range(REPEATS):
        for q in queries:
            t = time.perf_counter()
            emb @ q
            times.append(time.perf_counter() - t)
    return 1000 * np.array(times)


def batch_ms_per_query(emb, queries):
    matrix = np.stack(queries)
    matrix @ emb.T
    times = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        scores = matrix @ emb.T
        np.argpartition(-scores, DEPTH, axis=1)[:, :DEPTH]
        times.append(time.perf_counter() - t)
    return 1000 * min(times) / len(queries)


def main():
    corpus = load_corpus()
    texts = load_queries()
    val = load_splits().val
    rows = []
    print(f"cores: {os.cpu_count()}, numpy {np.__version__}, queries {len(val)}, depth {DEPTH}")
    for name, model_name, prefix in (("MiniLM (384)", DEFAULT_MODEL, ""), ("bge-base (768)", BGE, BGE_PREFIX)):
        model = SentenceTransformer(model_name, device="cpu")
        _, base, _ = encode_corpus(model, corpus, model_name)
        base = np.ascontiguousarray(base, dtype=np.float32)
        queries = [model.encode(prefix + texts[q], normalize_embeddings=True, show_progress_bar=False).astype(np.float32) for q in val]
        t = []
        for q in val[:50]:
            s = time.perf_counter()
            model.encode(prefix + texts[q], normalize_embeddings=True, show_progress_bar=False)
            t.append(time.perf_counter() - s)
        print(f"\n{name}: query encoding median {1000 * np.median(t):.1f} ms")
        for q in queries[:20]:
            scores = base @ q
            assert set(top_sorted(scores)) == set(top_partition(scores)), "argpartition and argsort disagree on the base corpus"
        for copies in COPIES:
            emb = np.ascontiguousarray(np.tile(base, (copies, 1)))
            row = {"model": name, "documents": len(emb), "size_mb": emb.nbytes / 1e6}
            for label, fn in (("matvec", None), ("matvec + full sort", top_sorted), ("matvec + partition", top_partition)):
                ms = matvec_ms(emb, queries) if fn is None else per_query_ms(fn, emb, queries)
                row[f"{label} median ms"] = float(np.median(ms))
                row[f"{label} p95 ms"] = float(np.quantile(ms, 0.95))
            row["batch ms per query"] = batch_ms_per_query(emb, queries)
            rows.append(row)
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})
            del emb
    pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
