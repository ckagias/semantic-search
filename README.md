# Semantic Search

Keyword search (BM25), pretrained embeddings and a fine-tuned bi-encoder, compared on BEIR SciFact with the same metrics and the same queries. The aim is to understand retrieval end to end: index, query, rank, metrics, and contrastive fine-tuning. A tie with BM25 is a valid result if it can be explained.

Status: work in progress. Phases 0 to 5 are done (setup, data, evaluation, BM25, pretrained embeddings, fine-tuning). Error analysis, scaling and the API are still to do. All numbers below are validation numbers (165 queries); the 300 test queries have not been scored yet.

## Results so far

nDCG@10 on the 165 validation queries, 95% bootstrap interval over queries. Full write-up in [results.md](results.md).

| system | nDCG@10 | recall@10 | recall@100 |
|---|---|---|---|
| Random ranking | 0.000 to 0.004 | 0.000 to 0.012 | 0.018 to 0.054 |
| BM25 (tuned, Snowball stemming) | 0.698 (0.637 to 0.757) | 0.829 | 0.929 |
| all-MiniLM-L6-v2, pretrained | 0.625 (0.563 to 0.687) | 0.760 | 0.943 |
| all-MiniLM-L6-v2, fine-tuned, in-batch negatives | 0.631 (0.568 to 0.692) | 0.771 | 0.942 |
| all-MiniLM-L6-v2, fine-tuned, BM25 hard negatives | 0.662 (0.601 to 0.722) | 0.793 | 0.898 |
| bge-small-en-v1.5, pretrained | 0.724 (0.668 to 0.780) | n/a | 0.968 |
| bge-base-en-v1.5, pretrained | 0.759 (0.705 to 0.810) | n/a | 0.974 |
| Hybrid: BM25 + bge-base, weighted sum (alpha 0.2) | 0.794 (0.742 to 0.842) | 0.908 | 0.968 |

- Pretrained bge-base beats stemmed BM25 by +0.061 (+0.017 to +0.106); MiniLM and mpnet-base lose to it.
- The hybrid beats both parts: +0.096 over BM25 and +0.034 over bge-base alone.
- Fine-tuning MiniLM on 475 train pairs: in-batch negatives alone give no measurable gain (+0.006); BM25 hard negatives give +0.036 (+0.011 to +0.062) but stay below BM25 and cost recall@100.
- Selection on the same validation queries makes the best rows slightly optimistic.

## Layout

```
src/
  data.py, splits.py          load SciFact; group-aware train/validation split, leaky-query flag
  metrics.py, evaluate.py     recall@k, MRR@10, nDCG@10, per-query scores, bootstrap intervals
  baselines.py                random and shortest-first floors
  index.py, keyword.py        tokenizer, inverted index, TF-IDF and BM25 by hand
  stemming.py                 Snowball stemming for BM25
  dense.py                    corpus encoding with cached embeddings, exact cosine search
  hybrid.py                   weighted-sum and reciprocal-rank fusion of BM25 and dense scores
  pairs.py, loss.py, train.py fine-tuning pairs, in-batch contrastive loss by hand, training loop
  tune_bm25.py, score_*.py, compare_queries.py, summarize_finetune.py   experiment scripts
tests/                        metrics worked by hand, BM25 against rank_bm25, loss against sentence-transformers, split and pair leakage checks
```

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Data: BEIR SciFact (`scifact.zip` from the BEIR dataset mirror), unpacked so that `data/scifact/` holds `corpus.jsonl`, `queries.jsonl` and `qrels/{train,test}.tsv`. Torch is the CPU build; every experiment here runs on the CPU.

## Running

```
python -m pytest
python -m src.score_bm25                    # tuned stemmed BM25 on validation
python -m src.score_dense BAAI/bge-base-en-v1.5
python -m src.score_hybrid
python -m src.train --negatives hard        # fine-tune MiniLM; --n-queries and --seed for the ablations
python -m src.summarize_finetune
```

Scripts write per-query score tables to `data/` and models to `models/`; both are untracked.

## Rules followed

- One evaluation function for every system, tested by hand before any model was scored.
- Settings chosen on validation queries only. Test queries are scored once per final system.
- Fine-tuning uses train queries that share no relevant document with a test query (415 of 644 train queries).
