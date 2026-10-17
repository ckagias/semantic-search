import time

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.data import DATA_DIR, load_corpus, load_qrels, load_queries
from src.dense import encode_corpus
from src.evaluate import evaluate
from src.score_dense import BGE_PREFIX
from src.splits import load_splits

MODEL = "BAAI/bge-base-en-v1.5"
K = 10
COPIES = 100
NOISE = 0.02
IVF_SMALL = (64, [1, 2, 4, 8, 16])
IVF_LARGE = (2048, [1, 4, 16, 64])
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 100
EF = [16, 32, 64, 128, 256, 512, 1024]
OUT = DATA_DIR.parent / "bench_ann.csv"


def synthetic_corpus(base, copies, noise, seed=0):
    rng = np.random.default_rng(seed)
    parts = [base]
    for _ in range(copies - 1):
        x = base + rng.normal(0, noise, base.shape).astype(np.float32)
        parts.append(x / np.linalg.norm(x, axis=1, keepdims=True))
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)


def exact_top(emb, queries, k, chunk=100_000):
    best_s = np.full((len(queries), k), -np.inf, dtype=np.float32)
    best_i = np.zeros((len(queries), k), dtype=np.int64)
    for start in range(0, len(emb), chunk):
        scores = queries @ emb[start:start + chunk].T
        idx = np.argpartition(-scores, k, axis=1)[:, :k]
        s = np.take_along_axis(scores, idx, axis=1)
        all_s = np.concatenate([best_s, s], axis=1)
        all_i = np.concatenate([best_i, idx + start], axis=1)
        keep = np.argsort(-all_s, axis=1)[:, :k]
        best_s = np.take_along_axis(all_s, keep, axis=1)
        best_i = np.take_along_axis(all_i, keep, axis=1)
    return best_i, best_s


def timed_search(index, queries, k):
    index.search(queries[:1], k)
    ids, scores, times = [], [], []
    for q in queries:
        t = time.perf_counter()
        s, i = index.search(q[None], k)
        times.append(time.perf_counter() - t)
        ids.append(i[0])
        scores.append(s[0])
    return np.array(ids), np.array(scores), 1000 * np.array(times)


def overlap(found, truth, modulo=None):
    if modulo:
        found, truth = found % modulo, truth % modulo
    return float(np.mean([len(set(a) & set(b)) / len(set(b)) for a, b in zip(found, truth)]))


def build_indexes(emb, ivf, d):
    out = {}
    t = time.perf_counter()
    flat = faiss.IndexFlatIP(d)
    flat.add(emb)
    out["flat (exact)"] = (flat, time.perf_counter() - t, [None])
    nlist, probes = ivf
    t = time.perf_counter()
    quant = faiss.IndexFlatIP(d)
    ivf_index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(0)
    sample = emb[rng.choice(len(emb), min(len(emb), 100_000), replace=False)]
    ivf_index.train(sample)
    ivf_index.add(emb)
    out[f"IVF nlist {nlist}"] = (ivf_index, time.perf_counter() - t, probes)
    t = time.perf_counter()
    hnsw = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    hnsw.add(emb)
    out[f"HNSW M {HNSW_M}"] = (hnsw, time.perf_counter() - t, EF)
    return out


def set_param(index, name, value):
    if value is None:
        return
    if name.startswith("IVF"):
        index.nprobe = value
    else:
        index.hnsw.efSearch = value


def main():
    corpus = load_corpus()
    texts = load_queries()
    qrels = load_qrels("train")
    val = load_splits().val
    model = SentenceTransformer(MODEL, device="cpu")
    doc_ids, base, _ = encode_corpus(model, corpus, MODEL)
    base = np.ascontiguousarray(base, dtype=np.float32)
    queries = np.stack([model.encode(BGE_PREFIX + texts[q], normalize_embeddings=True, show_progress_bar=False) for q in val]).astype(np.float32)
    d = base.shape[1]
    rows = []

    print(f"\nSciFact, {len(base)} real vectors, exact top {K} as truth")
    truth_small, _ = exact_top(base, queries, K)
    for name, (index, build, params) in build_indexes(base, IVF_SMALL, d).items():
        for p in params:
            set_param(index, name, p)
            ids, _, ms = timed_search(index, queries, K)
            by_text = {texts[q]: i for q, i in zip(val, ids)}

            def run(query, by_text=by_text):
                return [doc_ids[i] for i in by_text[query]]

            result = evaluate(run, texts, qrels, val, n_boot=10)
            row = {"corpus": "SciFact real", "vectors": len(base), "index": name, "param": p, "build s": build, "median ms": float(np.median(ms)), "p95 ms": float(np.quantile(ms, 0.95)), f"overlap@{K}": overlap(ids, truth_small), "ndcg@10": float(result.per_query["ndcg@10"].mean()), "recall@10": float(result.per_query["recall@10"].mean())}
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

    emb = synthetic_corpus(base, COPIES, NOISE)
    sim = float(np.mean(np.sum(emb[len(base):2 * len(base)] * base, axis=1)))
    print(f"\nsynthetic corpus: {len(emb)} vectors ({COPIES} noisy copies, mean cosine copy to original {sim:.2f}), {emb.nbytes / 1e9:.2f} GB")
    t = time.perf_counter()
    truth, truth_scores = exact_top(emb, queries, K)
    print(f"exact ground truth in {time.perf_counter() - t:.0f} s")
    for name, (index, build, params) in build_indexes(emb, IVF_LARGE, d).items():
        for p in params:
            set_param(index, name, p)
            ids, scores, ms = timed_search(index, queries, K)
            row = {
                "corpus": "synthetic",
                "vectors": len(emb),
                "index": name,
                "param": p,
                "build s": build,
                "median ms": float(np.median(ms)),
                "p95 ms": float(np.quantile(ms, 0.95)),
                f"overlap@{K}": overlap(ids, truth),
                f"source-document overlap@{K}": overlap(ids, truth, len(base)),
                "score ratio": float(scores.sum() / truth_scores.sum()),
            }
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
        del index
    pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
