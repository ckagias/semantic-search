import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.compare_queries import first_rank
from src.data import DATA_DIR, load_corpus, load_qrels, load_queries
from src.dense import DenseSearch, encode_corpus
from src.index import doc_tokens, tokenize
from src.keyword import build_bm25
from src.splits import load_splits
from src.train import MODELS_DIR

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FT_RUN = "all-MiniLM-L6-v2_hard_s0"
FT_EPOCH = 3
SHOW = 10
BIG = 1000
OUT = DATA_DIR.parent / "error_analysis_val.csv"


def top1_title(ranking, corpus):
    return corpus[ranking[0]]["title"]


def main():
    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels("train")
    val = load_splits().val

    bm25 = build_bm25(corpus).system()
    base = SentenceTransformer(BASE_MODEL, device="cpu")
    doc_ids, base_emb, _ = encode_corpus(base, corpus, BASE_MODEL)
    pretrained = DenseSearch(base, doc_ids, base_emb).system()
    ft = SentenceTransformer(str(MODELS_DIR / FT_RUN), device="cpu")
    _, ft_emb, _ = encode_corpus(ft, corpus, f"ft_{FT_RUN}_epoch{FT_EPOCH}")
    tuned = DenseSearch(ft, doc_ids, ft_emb).system()

    rows = []
    for qid in val:
        relevant = set(qrels[qid])
        q_tokens = set(tokenize(queries[qid]))
        doc_vocab = set().union(*(doc_tokens(corpus[d]) for d in relevant))
        ranks = {"bm25": bm25(queries[qid]), "pretrained": pretrained(queries[qid]), "tuned": tuned(queries[qid])}
        rows.append(
            {
                "qid": qid,
                "query": queries[qid],
                "title": corpus[next(iter(relevant))]["title"],
                "overlap": len(q_tokens & doc_vocab) / len(q_tokens),
                **{f"{name}_rank": first_rank(r, relevant) for name, r in ranks.items()},
                "bm25_top1": top1_title(ranks["bm25"], corpus),
                "tuned_top1": top1_title(ranks["tuned"], corpus),
            }
        )
    df = pd.DataFrame(rows).set_index("qid")
    for name in ("bm25", "pretrained", "tuned"):
        df[f"{name}_rr"] = 1 / df[f"{name}_rank"].fillna(BIG)
    df["diff"] = df["tuned_rr"] - df["bm25_rr"]
    df.to_csv(OUT)

    check = pd.read_csv(DATA_DIR.parent / f"ft_{FT_RUN}_val_per_query.csv", index_col=0)
    print(f"fine-tuned MRR@100-style check: nDCG@10 from saved scores {check['ndcg@10'].mean():.3f}")
    print(f"tuned better than BM25 on {(df['diff'] > 0).sum()} queries, worse on {(df['diff'] < 0).sum()}, equal on {(df['diff'] == 0).sum()}")
    print(f"BM25 and tuned both rank 1: {((df['bm25_rank'] == 1) & (df['tuned_rank'] == 1)).sum()}, both miss top 100: {(df['bm25_rank'].isna() & df['tuned_rank'].isna()).sum()}")

    df["bucket"] = pd.qcut(df["overlap"], 3, labels=["low", "mid", "high"])
    print("\nby overlap tercile (share of query words found in the relevant document)")
    print(
        df.groupby("bucket", observed=True)
        .agg(
            n=("overlap", "size"),
            overlap=("overlap", "mean"),
            bm25_mrr=("bm25_rr", "mean"),
            pretrained_mrr=("pretrained_rr", "mean"),
            tuned_mrr=("tuned_rr", "mean"),
            tuned_wins=("diff", lambda x: (x > 0).sum()),
            bm25_wins=("diff", lambda x: (x < 0).sum()),
        )
        .round(3)
        .to_string()
    )
    print(f"\ncorrelation of (tuned - BM25) reciprocal rank with overlap: {np.corrcoef(df['diff'], df['overlap'])[0, 1]:.3f}")

    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    cols = ["query", "title", "bm25_rank", "pretrained_rank", "tuned_rank", "overlap"]
    print(f"\nBM25 beats tuned by most ({SHOW}):")
    print(df.sort_values("diff").head(SHOW)[cols].round(2).to_string())
    print("\ntop result of tuned for those queries:")
    for qid, row in df.sort_values("diff").head(SHOW).iterrows():
        print(f"  {qid}: {row['tuned_top1']}")
    print(f"\ntuned beats BM25 by most ({SHOW}):")
    print(df.sort_values("diff", ascending=False).head(SHOW)[cols].round(2).to_string())
    print("\ntop result of BM25 for those queries:")
    for qid, row in df.sort_values("diff", ascending=False).head(SHOW).iterrows():
        print(f"  {qid}: {row['bm25_top1']}")


if __name__ == "__main__":
    main()
