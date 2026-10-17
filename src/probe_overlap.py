import random

import nltk
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.data import DATA_DIR, load_corpus, load_qrels, load_queries
from src.dense import DenseSearch, encode_corpus
from src.evaluate import bootstrap_ci, evaluate
from src.index import InvertedIndex, tokenize, tokenize_stemmed
from src.keyword import BM25_TUNED, BM25
from src.score_dense import BGE_PREFIX, DEFAULT_MODEL
from src.splits import load_splits
from src.train import MODELS_DIR

BGE = "BAAI/bge-base-en-v1.5"
FT_RUN = "all-MiniLM-L6-v2_hard_s0"
FT_EPOCH = 3
RARE_FRACTION = 0.10
MIN_WORDS = 3
MIN_SYNONYM_LENGTH = 4
RANDOM_DRAWS = 5
OUT = DATA_DIR.parent / "probe_overlap_val_per_query.csv"


def stem(word):
    return tokenize_stemmed(word)[0]


def overlapping_words(tokens, doc_stems, rare):
    return [i for i, t in enumerate(tokens) if stem(t) in doc_stems and stem(t) in rare]


def delete_words(tokens, positions):
    drop = set(positions)
    return [t for i, t in enumerate(tokens) if i not in drop]


def synonym(word, banned_stems):
    from nltk.corpus import wordnet

    if len(word) < MIN_SYNONYM_LENGTH or not word.isalpha():
        return None
    counts = {}
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            name = lemma.name().lower()
            if "_" in name or "-" in name or not name.isalpha() or name == word:
                continue
            if stem(name) == stem(word) or stem(name) in banned_stems:
                continue
            counts[name] = counts.get(name, 0) + lemma.count()
    if not counts:
        return None
    return sorted(counts, key=lambda n: (-counts[n], n))[0]


def replace_words(tokens, positions, doc_stems):
    out, replaced = list(tokens), 0
    drop = set()
    for i in positions:
        new = synonym(tokens[i], doc_stems)
        if new is None:
            drop.add(i)
        else:
            out[i] = new
            replaced += 1
    return [t for i, t in enumerate(out) if i not in drop], replaced


def build_variants(qid, text, relevant_docs, corpus, df, n_docs):
    rare = {t for t, v in df.items() if v <= RARE_FRACTION * n_docs}
    tokens = tokenize(text)
    doc_stems = set()
    for d in relevant_docs:
        doc_stems.update(tokenize_stemmed(corpus[d]["title"] + " " + corpus[d]["text"]))
    positions = overlapping_words(tokens, doc_stems, rare)
    if not positions or len(tokens) - len(positions) < MIN_WORDS:
        return None
    rare_positions = [i for i, t in enumerate(tokens) if stem(t) in rare]
    variants = {"control": " ".join(tokens), "overlap_deleted": " ".join(delete_words(tokens, positions))}
    replaced_tokens, n_replaced = replace_words(tokens, positions, doc_stems)
    variants["overlap_replaced"] = " ".join(replaced_tokens)
    rarest = min(positions, key=lambda i: (df[stem(tokens[i])], i))
    variants["rarest_deleted"] = " ".join(delete_words(tokens, [rarest]))
    for draw in range(RANDOM_DRAWS):
        rng = random.Random(f"{qid}-{draw}")
        variants[f"random_{draw}"] = " ".join(delete_words(tokens, rng.sample(rare_positions, len(positions))))
        variants[f"random1_{draw}"] = " ".join(delete_words(tokens, rng.sample(rare_positions, 1)))
    return variants, len(positions), n_replaced, len(tokens)


def main():
    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels("train")
    val = load_splits().val
    index = InvertedIndex(corpus, tokenize_stemmed)
    df = {t: len(p) for t, p in index.postings.items()}

    variants, meta = {}, {}
    for qid in val:
        built = build_variants(qid, queries[qid], list(qrels[qid]), corpus, df, len(corpus))
        if built:
            variants[qid], *meta[qid] = built
    kept = list(variants)
    removed = sum(m[0] for m in meta.values())
    replaced = sum(m[1] for m in meta.values())
    print(f"{len(kept)} of {len(val)} validation queries have a removable overlapping word and at least {MIN_WORDS} words left")
    print(f"overlapping rare words removed: {removed}, of which a WordNet synonym was found for {replaced}")

    systems = {"BM25": BM25(index, **BM25_TUNED).system()}
    for name, model_name, prefix, path in [
        ("MiniLM pretrained", DEFAULT_MODEL, "", None),
        ("MiniLM fine-tuned", None, "", MODELS_DIR / FT_RUN),
        ("bge-base", BGE, BGE_PREFIX, None),
    ]:
        model = SentenceTransformer(str(path or model_name), device="cpu")
        cache_name = f"ft_{FT_RUN}_epoch{FT_EPOCH}" if path else model_name
        doc_ids, emb, _ = encode_corpus(model, corpus, cache_name)
        systems[name] = DenseSearch(model, doc_ids, emb, prefix).system()

    rows = []
    for name, system in systems.items():
        per = {}
        conds = ["control", "overlap_deleted", "overlap_replaced", "rarest_deleted"]
        conds += [f"random_{d}" for d in range(RANDOM_DRAWS)] + [f"random1_{d}" for d in range(RANDOM_DRAWS)]
        for cond in conds:
            qs = {qid: variants[qid][cond] for qid in kept}
            per[cond] = evaluate(system, qs, qrels, kept, n_boot=10).per_query
        for qid in kept:
            rows.append(
                {
                    "system": name,
                    "qid": qid,
                    **{f"{c}_ndcg": per[c].loc[qid, "ndcg@10"] for c in ("control", "overlap_deleted", "overlap_replaced", "rarest_deleted")},
                    "random_ndcg": float(np.mean([per[f"random_{d}"].loc[qid, "ndcg@10"] for d in range(RANDOM_DRAWS)])),
                    "random1_ndcg": float(np.mean([per[f"random1_{d}"].loc[qid, "ndcg@10"] for d in range(RANDOM_DRAWS)])),
                    "control_recall100": per["control"].loc[qid, "recall@100"],
                    "overlap_deleted_recall100": per["overlap_deleted"].loc[qid, "recall@100"],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("\nnDCG@10 on the probed queries (mean), and drop against the control (paired 95% interval)")
    for name in systems:
        d = df[df["system"] == name]
        line = [f"\n{name:18s} control {d['control_ndcg'].mean():.3f}"]
        for label, col in (
            ("random deletion", "random_ndcg"),
            ("overlap deleted", "overlap_deleted_ndcg"),
            ("overlap replaced", "overlap_replaced_ndcg"),
            ("one random rare word deleted", "random1_ndcg"),
            ("rarest overlapping word deleted", "rarest_deleted_ndcg"),
        ):
            mean, lo, hi = bootstrap_ci(d[col] - d["control_ndcg"])
            line.append(f"{label} {d[col].mean():.3f} ({mean:+.3f}; {lo:+.3f} to {hi:+.3f})")
        print(" | ".join(line))
    print("\nrecall@100 control -> overlap deleted")
    for name in systems:
        d = df[df["system"] == name]
        print(f"{name:18s} {d['control_recall100'].mean():.3f} -> {d['overlap_deleted_recall100'].mean():.3f}")


if __name__ == "__main__":
    nltk.download("wordnet", quiet=True)
    main()
