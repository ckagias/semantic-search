# Results

Dataset: BEIR SciFact, 5,183 documents (title + abstract), binary relevance, mean 1.14 relevant documents per train query. All scores are over the 165 validation queries unless stated (the Test scores section is the exception: it holds the only scoring of the 300 test queries); intervals are 95% bootstrap intervals over queries (10,000 resamples), paired when comparing two systems on the same queries. The 300 test queries were scored once, at the end (see Test scores). Fine-tuning figures are for seed 0 unless a seed range is stated; the range over seeds 0 to 2 is in the Seeds section.

## Splits

644 train, 165 validation, 300 test queries. Train queries are grouped by shared relevant documents (490 groups) and whole groups move to validation, so no relevant document is shared between train and validation. 272 of the 809 labelled queries have a relevant document that also belongs to a test query; these are excluded from fine-tuning, which leaves 415 clean train queries (475 pairs, 292 unique documents). Tests assert that no clean pair contains a validation or test query or a test-relevant document.

## Floors

Random ranking: nDCG@10 0.000 to 0.004 over 5 seeds. Shortest document first: nDCG@10 0.000, recall@100 0.006. Document length carries no signal.

## Keyword search

BM25 and TF-IDF are implemented from an inverted index. BM25 with the rank_bm25 idf matches `rank_bm25.BM25Okapi` on all validation queries (largest absolute difference 4e-12). The rank_bm25 idf gives every term present in more than half the documents the same weight (1.81), so "the" weighs as much as "cancer"; the Lucene idf, ln(1 + (N - df + 0.5) / (df + 0.5)), does not.

Tuning (Lucene idf, k1 and b grid, 198 settings): the surface is a plateau, with the ten best settings within 0.003 nDCG@10. Snowball stemming adds +0.046 (+0.021 to +0.073), better on 34 queries and worse on 16.

| BM25 | nDCG@10 | MRR@10 | recall@10 | recall@100 |
|---|---|---|---|---|
| no stemming (k1 2.1, b 0.5) | 0.651 (0.589 to 0.713) | 0.614 | 0.796 | 0.906 |
| Snowball stemming (k1 1.8, b 0.5) | 0.698 (0.637 to 0.757) | 0.663 | 0.829 | 0.929 |

The stemmed, tuned BM25 is the baseline every later system is compared with. 98 of 165 queries have the answer at rank 1; 11 have none in the top 100.

## Pretrained embeddings

Exact cosine search over normalised embeddings, title + abstract, CPU. BGE models use the retrieval query prefix.

| model | params | max tokens | corpus encoding | nDCG@10 | recall@100 | vs stemmed BM25 (paired) |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 22M | 256 | 98 s | 0.625 (0.563 to 0.687) | 0.943 | -0.072 (-0.122 to -0.020) |
| all-mpnet-base-v2 | 110M | 384 | 1,290 s | 0.622 (0.558 to 0.687) | 0.927 | -0.075 (-0.126 to -0.025) |
| bge-small-en-v1.5 | 33M | 512 | 315 s | 0.724 (0.668 to 0.780) | 0.968 | +0.027 (-0.017 to +0.071) |
| bge-base-en-v1.5 | 110M | 512 | 1,053 s | 0.759 (0.705 to 0.810) | 0.974 | +0.061 (+0.017 to +0.106) |

Size does not explain the ranking: mpnet-base is five times the size of MiniLM and scores the same. Max sequence length, training recipe and the query prefix differ between the families and were not separated. Four models were compared on the same queries, so the best margin is slightly optimistic. Overlap between the BGE training data and SciFact was not checked.

Against unstemmed BM25, the gain of bge-base over BM25 grows as the query shares fewer words with its relevant document: on the lowest overlap tercile MRR is 0.491 against 0.270, on the highest 0.980 against 0.973. BM25 wins mainly on queries with rare exact names. This comparison predates stemming and is to be repeated against the stemmed baseline.

## Hybrid

Stemmed BM25 and bge-base fused by a weighted sum of min-max normalised scores (weight alpha on BM25) or by reciprocal rank fusion (k in {10, 60, 200}).

| system | nDCG@10 | MRR@10 | recall@10 | recall@100 |
|---|---|---|---|---|
| weighted sum, alpha 0.2 | 0.794 (0.742 to 0.842) | 0.764 | 0.908 | 0.968 |
| bge-base alone | 0.759 | n/a | n/a | 0.974 |
| BM25 alone | 0.698 | 0.663 | 0.829 | 0.929 |

Against BM25: +0.096 (+0.063 to +0.130), better on 50 queries, worse on 8. Against bge-base alone: +0.034 (+0.010 to +0.059), better on 32, worse on 16. Every fusion variant is flat between alpha 0.1 and 0.5 (0.76 to 0.79). 44 settings were tried on the same queries, so the maximum is optimistic; RRF without a fitted score scale gives 0.78.

## Fine-tuning

all-MiniLM-L6-v2, 475 clean pairs. The in-batch contrastive loss (cosine similarity times 20, cross-entropy against the diagonal) is written by hand and matches `MultipleNegativesRankingLoss` of sentence-transformers 6.1.0 in value and gradients (tolerance 1e-5 and 1e-6). Batches of up to 16 contain no repeated query or document, and a pair's hard negative is never another pair's positive in the same batch. AdamW, learning rate 2e-5, weight decay 0.01, 10% warmup then linear decay, 4 epochs; the full corpus is re-encoded and scored on validation after every epoch and the best epoch is kept. About 60 s (in-batch) or 100 s (hard negatives) of training and 100 s of validation per epoch on the CPU.

Hard negatives: per pair, one document drawn at random from the top 10 stemmed-BM25 results that are not judged relevant to the query and not relevant to any test query.

| setting | best epoch | nDCG@10 | recall@10 | recall@100 |
|---|---|---|---|---|
| pretrained | n/a | 0.625 (0.563 to 0.687) | 0.760 | 0.943 |
| in-batch negatives | 4 | 0.631 (0.568 to 0.692) | 0.771 | 0.942 |
| hard negatives | 3 | 0.662 (0.601 to 0.722) | 0.793 | 0.898 |

Paired nDCG@10 differences: hard negatives minus pretrained +0.036 (+0.011 to +0.062); hard negatives minus in-batch +0.031 (+0.011 to +0.053); in-batch minus pretrained +0.006 (-0.011 to +0.022); hard negatives minus stemmed BM25 -0.036 (-0.089 to +0.020).

- In-batch negatives alone: training loss falls from 0.215 to 0.07 within two epochs. The validation gain is small and depends on the seed (+0.006 to +0.024, see Seeds), as the 15 random negatives are mostly easy.
- Hard negatives: the gain appears from epoch 2 and is flat over epochs 2 to 4 (0.659 to 0.662). recall@10 rises (0.760 to 0.793) while recall@100 falls (0.943 to about 0.90).
- The fine-tuned MiniLM does not reach BM25, bge-small or bge-base.

### Ablation on the number of train queries (hard negatives, seed 0; three seeds below)

Queries are a seeded sample of the 415 clean train queries. 400 was replaced by 200 because only 415 are usable. Update steps scale with the data (28, 56 and 120), so queries and steps change together.

| train queries | pairs | best nDCG@10 | recall@10 | recall@100 |
|---|---|---|---|---|
| 0 (pretrained) | 0 | 0.625 | 0.760 | 0.943 |
| 100 | 111 | 0.631 | 0.768 | 0.934 |
| 200 | 219 | 0.656 | 0.805 | 0.934 |
| 415 | 475 | 0.662 | 0.793 | 0.898 |

### Seeds

Seeds 0, 1 and 2 for every setting. The seed changes the batch order, the choice of hard negatives and, in the ablation, which queries are sampled. Best-epoch validation nDCG@10 per run (pretrained 0.625):

| setting | seed 0 | seed 1 | seed 2 | mean | recall@10 (mean) | recall@100 (mean) |
|---|---|---|---|---|---|---|
| in-batch negatives, 415 queries | 0.631 | 0.639 | 0.650 | 0.640 | 0.783 | 0.944 |
| hard negatives, 100 queries | 0.631 | 0.635 | 0.642 | 0.636 | 0.770 | 0.929 |
| hard negatives, 200 queries | 0.656 | 0.648 | 0.653 | 0.652 | 0.811 | 0.928 |
| hard negatives, 415 queries | 0.662 | 0.661 | 0.662 | 0.662 | 0.811 | 0.908 |

Paired nDCG@10 differences on the validation queries (95% interval), per seed:

| comparison | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| hard negatives minus pretrained | +0.036 (+0.011 to +0.062) | +0.036 (+0.010 to +0.062) | +0.037 (+0.015 to +0.058) |
| in-batch minus pretrained | +0.006 (-0.011 to +0.022) | +0.014 (-0.002 to +0.029) | +0.024 (+0.007 to +0.041) |
| hard negatives minus in-batch | +0.031 (+0.011 to +0.053) | +0.022 (0.000 to +0.045) | +0.012 (-0.006 to +0.030) |
| hard negatives minus stemmed BM25 | -0.036 (-0.089 to +0.020) | -0.037 (-0.088 to +0.017) | -0.036 (-0.086 to +0.015) |

- Hard negatives with all queries are stable across seeds (0.661 to 0.662) and always beat the pretrained model by about +0.036. They never reach BM25 (-0.036 in every seed, intervals include zero).
- In-batch negatives are not a null result: the gain over pretrained is +0.006, +0.014 and +0.024 depending on the seed, so seed 0 understated it. The in-batch runs spread over 0.019, the hard-negative runs over 0.001.
- Hard negatives beat in-batch in all three seeds, by +0.012 to +0.031 (mean +0.022), but the per-seed interval excludes zero only for seed 0 (seed 1 touches zero, seed 2 includes it). The ordering is consistent; its size is not well pinned down.
- The best epoch is chosen per run on the same validation queries, which adds a little optimism to every row equally. The best epoch of the hard-negative runs was 3 or 4, of the in-batch runs 2 to 4.
- Data size: 100 queries give +0.011 (0.636), 200 give +0.027 (0.652) and 415 give +0.037 (0.662) over pretrained. The spread between seeds is at most 0.011 at 100 queries and 0.001 at 415.
- With all queries, recall@100 falls to 0.908 with hard negatives (0.898 to 0.922) while in-batch keeps 0.944.

## Error analysis: fine-tuned MiniLM against stemmed BM25 (validation, `src/analyze_errors.py`)

Fine-tuned model: MiniLM, hard negatives, all clean queries, seed 0, epoch 3 (nDCG@10 0.662). Reciprocal rank is 1 / rank of the first relevant document over the full ranking of the corpus (1/1000 when it is outside the top 100). Overlap is the share of a query's distinct words that occur in its relevant document. Per-query table in `data/error_analysis_val.csv`.

Rank of the first relevant document (165 queries):

| system | rank 1 | rank 2 to 10 | rank 11 to 100 | not in top 100 |
|---|---|---|---|---|
| stemmed BM25 | 98 | 42 | 14 | 11 |
| MiniLM pretrained | 79 | 48 | 29 | 9 |
| MiniLM fine-tuned | 88 | 44 | 17 | 16 |

Fine-tuned against BM25: better on 39 queries, worse on 46, equal on 80 (70 of them with both at rank 1, 5 with both outside the top 100). Of the 46 losses, 23 still have the fine-tuned model in the top 10 and 28 have BM25 at rank 1; 7 are large (BM25 in the top 3, fine-tuned outside the top 10). Of the 39 wins, 16 have BM25 outside the top 10 and 6 are large (fine-tuned in the top 3, BM25 outside the top 10).

By keyword overlap (terciles of the overlap):

| overlap tercile | queries | mean overlap | BM25 MRR | pretrained MRR | fine-tuned MRR | fine-tuned better | BM25 better |
|---|---|---|---|---|---|---|---|
| low | 63 | 0.35 | 0.340 | 0.343 | 0.413 | 29 | 21 |
| mid | 53 | 0.61 | 0.766 | 0.644 | 0.661 | 9 | 17 |
| high | 49 | 0.80 | 0.976 | 0.860 | 0.873 | 1 | 8 |

- The fine-tuned model is ahead only on the low-overlap third. In the high-overlap third, where the document repeats most of the query's words, BM25 is near perfect (0.976) and the fine-tuned model is not (0.873, +0.013 over pretrained). Fine-tuning raised all three terciles, most on the low one (+0.070). The correlation between (fine-tuned minus BM25) reciprocal rank and overlap is -0.22.
- The 10 largest BM25 wins were read. In 8, the fine-tuned top result is a paper on the same topic that is not the one the claim comes from (panic anxiety returns a prevalence review for the orexin paper; H. pylori and gastric cancer returns a mouse H. pylori mutation paper; bariatric surgery and colorectal cancer returns a colorectal cancer survival paper). In 2, the claim hinges on a rare name (HOXB4; Esrrb with Mbd3), BM25 puts the paper at rank 1, and the fine-tuned model returns an unrelated paper and misses the top 100.
- The 10 largest fine-tuned wins were read. In 8, the claim and the document use different words for the same thing: students against pupils (bullying), OCT3/4 against Oct4, a drug name against the transport process it inhibits (AM404 and anandamide), "neural progenitors differ across species" against "species-specific differences in progenitor behavior". In 6 of the 10, BM25's top result is off topic and matches only a few words of the claim ("pioneer factor" gives a Pax7 paper; the negated "no known interaction between Deltex and eIF3" gives a vitamin D paper); in the other 4, BM25 has the right topic but ranks the paper 4 to 9.
- Claims that differ only in direction (male/female swapped, positively/negatively related, "interacts" against "no known interaction") get the same rank for the same document in all three pairs found in validation. In SciFact the relevant paper is the same for both polarities, so this is not penalised here.
- Fine-tuning moves 9 queries to rank 1 relative to pretrained but pushes 7 more queries out of the top 100 (9 to 16), the same effect as the lower recall@100 above. The queries were picked by size of the rank gap, so rank-1 BM25 results are over-represented among the losses by construction.
- One model, one seed, 165 queries; the 20 queries were read by one person. The bge-base comparison against stemmed BM25 by overlap tercile (above, "Per-query comparison", run before stemming) has not been repeated.

## Keyword-overlap probe (validation, `src/probe_overlap.py`)

An informative overlapping word is a query word whose stem occurs in a relevant document and in at most 10% of the corpus (stemmed). 156 of 165 validation queries have at least one and keep at least three words after deleting them; 624 words are involved (4.0 per query), 9.2 words remain on average. Variants of each query (lowercased tokens rejoined; the control is the same tokens with nothing removed):

- overlap deleted: all informative overlapping words removed.
- overlap replaced: each is replaced by its most frequent WordNet synonym when one exists (at least four letters, alphabetic, single word, stem not in the relevant document) and deleted otherwise; a synonym was found for 265 of the 624 words.
- random deletion: the same number of informative words chosen at random from the query, mean of 5 draws.
- rarest deleted: only the overlapping word with the lowest document frequency; random-1: one random informative word, mean of 5 draws.

nDCG@10 on the 156 probed queries (change against the control in brackets):

| system | control | random deletion | overlap deleted | overlap replaced | rarest deleted | random-1 |
|---|---|---|---|---|---|---|
| stemmed BM25 | 0.729 | 0.405 (-0.324) | 0.023 (-0.705) | 0.010 (-0.718) | 0.529 (-0.199) | 0.668 (-0.061) |
| MiniLM pretrained | 0.661 | 0.311 (-0.350) | 0.094 (-0.567) | 0.161 (-0.500) | 0.473 (-0.188) | 0.598 (-0.063) |
| MiniLM fine-tuned | 0.694 | 0.338 (-0.356) | 0.097 (-0.597) | 0.171 (-0.523) | 0.493 (-0.201) | 0.644 (-0.050) |
| bge-base | 0.788 | 0.452 (-0.337) | 0.167 (-0.622) | 0.299 (-0.489) | 0.618 (-0.171) | 0.738 (-0.051) |

recall@100, control to overlap deleted: BM25 0.971 to 0.182, MiniLM pretrained 0.966 to 0.419, MiniLM fine-tuned 0.938 to 0.457, bge-base 0.985 to 0.601.

- With every informative overlapping word deleted all four systems lose most of their score: BM25 keeps 3% of its control nDCG@10, MiniLM 14% (pretrained and fine-tuned), bge-base 21%. The neural models fall less than BM25 (drop difference against BM25: pretrained MiniLM +0.138 (+0.079 to +0.197), fine-tuned +0.108 (+0.049 to +0.167), bge-base +0.083 (+0.025 to +0.142)), and keep more of recall@100, but they do not work without those words.
- Which words are removed matters more than how many: deleting the same number of random informative words costs 0.32 to 0.36 for every system, BM25 and neural alike.
- Deleting only the rarest overlapping word costs 0.17 to 0.20 for every system; the drop of the neural models against BM25 differs by at most 0.03 and every interval includes zero. One random informative word costs 0.05 to 0.06. The neural models are as sensitive as BM25 to losing the single most distinctive word.
- Synonym replacement recovers part of the loss for the neural models (+0.067 pretrained, +0.074 fine-tuned, +0.133 bge-base over deletion, all intervals above zero) and none for BM25 (0.023 to 0.010), but all stay well below the control. WordNet is general English and covers fewer than half of the words; replacements are often the wrong sense for biomedical text, so this condition understates what a model that matches meaning could do.
- Fine-tuning does not change robustness. Fine-tuned minus pretrained is +0.033 (+0.006 to +0.060) on the control, +0.002 (-0.013 to +0.018) with the overlap deleted and +0.009 (-0.007 to +0.026) with synonyms; the drop caused by deletion is 0.030 larger for the fine-tuned model (-0.060 to 0.000). The fine-tuning gain is not visible once the overlapping words are gone.
- The deleted words are the ones that identify the paper, so a large drop is unavoidable for any model and the probe cannot separate lexical matching from loss of information; the comparison with random deletion and between systems is what carries the result. One fine-tuned model, one seed, validation queries only.

## Ceiling for a re-ranker on top of each first stage (validation, `src/ceiling.py`)

For each first stage the top k documents are the candidates. A perfect re-ranker (oracle) puts every relevant candidate first, so its nDCG@10 is the best any re-ranker can reach from those candidates. Intervals are 95% bootstrap over the 165 queries. With 1.14 relevant documents per query the oracle nDCG@10 is almost equal to recall of the candidates, and the share of queries with at least one relevant candidate is within 0.01 of it.

Oracle nDCG@10 by number of candidates (the first stage's own nDCG@10 in the last column):

| first stage | k = 10 | k = 20 | k = 50 | k = 100 | own nDCG@10 |
|---|---|---|---|---|---|
| stemmed BM25 | 0.833 | 0.848 | 0.901 | 0.930 (0.889 to 0.967) | 0.698 |
| MiniLM pretrained | 0.763 | 0.849 | 0.913 | 0.944 (0.908 to 0.976) | 0.625 |
| MiniLM fine-tuned | 0.795 | 0.837 | 0.872 | 0.899 (0.852 to 0.942) | 0.662 |
| bge-base | 0.895 | 0.916 | 0.950 | 0.974 (0.949 to 0.994) | 0.759 |
| hybrid BM25 + bge-base | 0.910 | 0.934 | 0.950 | 0.968 (0.939 to 0.993) | 0.794 |
| BM25 and bge-base, top 50 each (mean 84 documents, range 65 to 98) | n/a | n/a | n/a | 0.962 (0.932 to 0.988) | n/a |

- Headroom for a re-ranker over the top 100 is large: +0.23 over BM25, +0.22 over bge-base and +0.17 over the hybrid. No re-ranker can exceed 0.968 on this first stage, and one limited to 20 candidates cannot exceed 0.934.
- The hybrid is ahead of bge-base alone only at small k (recall 0.908 against 0.894 at 10, 0.932 against 0.914 at 20); at 50 and 100 the two are equal within the intervals. The hybrid's benefit is a better top of the list, not more coverage.
- The fine-tuned MiniLM has a higher ceiling than the pretrained one only at k = 10 (0.795 against 0.763); at k = 20 they are equal within the intervals and at k = 50 and 100 it is lower than the pretrained model (0.872 against 0.913, 0.899 against 0.944), which is the recall@100 loss seen after fine-tuning. As a first stage for a re-ranker it is worse than the model it started from.
- Taking the union of BM25's and bge-base's top 50 gives 0.962, no better than bge-base's top 100 (0.974): merging the two candidate sets adds no coverage over bge-base alone.
- The cost of running a re-ranker over 50 or 100 candidates per query was not measured here.

## Test scores (300 test queries, scored once, `src/score_test.py`)

Settings were fixed on validation before this run: BM25 k1 1.8, b 0.5 with Snowball stemming; hybrid = weighted sum of min-max normalised scores, alpha 0.2 on BM25, with bge-base; fine-tuned MiniLM = hard negatives on the 415 clean train queries, 475 pairs, best validation epoch of each of three seeds (3, 4, 4). Before scoring, the saved models and cached embeddings were checked to reproduce the validation nDCG@10 of their logs (0.6617, 0.6611, 0.6620). The script refuses to run again while `data/test_per_query.csv` exists. 95% bootstrap intervals over the 300 queries.

| system | nDCG@10 | MRR@10 | recall@10 | recall@100 |
|---|---|---|---|---|
| random ranking (5 seeds) | 0.000 to 0.002 | n/a | n/a | 0.006 to 0.028 |
| stemmed BM25 | 0.686 (0.639 to 0.730) | 0.653 | 0.808 | 0.916 |
| MiniLM pretrained | 0.645 (0.599 to 0.689) | 0.605 | 0.783 | 0.925 |
| MiniLM fine-tuned, seed 0 | 0.668 (0.623 to 0.712) | 0.628 | 0.809 | 0.930 |
| MiniLM fine-tuned, seed 1 | 0.662 (0.617 to 0.707) | 0.619 | 0.809 | 0.927 |
| MiniLM fine-tuned, seed 2 | 0.652 (0.607 to 0.696) | 0.611 | 0.795 | 0.927 |
| bge-base pretrained | 0.740 (0.700 to 0.779) | 0.703 | 0.874 | 0.967 |
| hybrid BM25 + bge-base | 0.752 (0.711 to 0.791) | 0.720 | 0.875 | 0.980 |

Paired nDCG@10 differences on the same queries (fine-tuned = mean of the three seeds per query, 0.661):

| comparison | difference | better / worse |
|---|---|---|
| bge-base minus BM25 | +0.055 (+0.022 to +0.086) | 86 / 41 |
| hybrid minus BM25 | +0.067 (+0.040 to +0.093) | 83 / 23 |
| hybrid minus bge-base | +0.012 (-0.005 to +0.029) | 52 / 35 |
| MiniLM pretrained minus BM25 | -0.041 (-0.076 to -0.006) | 62 / 73 |
| fine-tuned MiniLM minus pretrained | +0.016 (+0.001 to +0.031) | 55 / 44 |
| fine-tuned MiniLM minus BM25 | -0.025 (-0.059 to +0.009) | 69 / 77 |

- The ranking of the systems matches validation: hybrid (0.752) and bge-base (0.740) on top, then BM25 (0.686), fine-tuned MiniLM (0.652 to 0.668) and pretrained MiniLM (0.645). The test scores are lower than validation for the hybrid (0.794 to 0.752) and bge-base (0.759 to 0.740) and slightly lower for BM25 (0.698 to 0.686); the fine-tuned MiniLM is unchanged (0.662 on validation, 0.661 mean of seeds on test) and the pretrained MiniLM is higher (0.625 to 0.645).
- bge-base beats BM25 with an interval above zero (+0.055). The hybrid beats BM25 by +0.067, but its edge over bge-base alone (+0.012) is not distinguishable from zero on the test queries, against +0.034 (+0.010 to +0.059) on validation; the alpha was fitted on the validation queries, which explains part of the larger validation gap.
- Fine-tuning gives +0.016 (+0.001 to +0.031) over pretrained MiniLM on the test queries, less than half of the +0.036 on validation (best epoch and model choice were made on validation). The seed range on the test queries is 0.652 to 0.668 (0.016), against 0.001 on validation, so the seed matters more here than the validation spread suggested.
- Fine-tuned MiniLM against BM25: -0.025 (-0.059 to +0.009), as on validation not clearly below and not above. Fine-tuned MiniLM against pretrained bge-small or bge-base was not scored on test.
- recall@100 does not fall with fine-tuning on the test queries (0.927 to 0.930 against 0.925 pretrained), where it fell from 0.943 to 0.908 on validation. The loss seen on validation did not reproduce; the recall@100 difference was not tested on the test queries.
- Overlap between splits remains: 196 of the 300 test queries have a relevant document that a train query also points to. Fine-tuning used only the 415 clean train queries, so the models did not see those documents.

## Exact search as the corpus grows (`src/bench_scale.py`)

The SciFact embeddings are tiled 1, 10 and 100 times (5,183, 51,830 and 518,300 vectors, exact copies, float32). One query is one matrix-vector product against all vectors followed by the top 100. Median and 95th percentile over the 165 validation queries, 3 repeats, one process on 16 logical cores of a WSL2 machine, numpy default threads. The copies make the timing realistic for memory and arithmetic but not for ranking quality. Times are milliseconds per query.

| model | vectors | matrix | matvec | + full sort (current code) | + partition | batch of 165 (per query) |
|---|---|---|---|---|---|---|
| MiniLM, 384 | 5,183 | 8 MB | 0.02 | 0.36 (p95 1.3) | 0.10 | 0.08 |
| MiniLM, 384 | 51,830 | 80 MB | 0.19 | 5.0 (p95 7.6) | 1.1 | 0.80 |
| MiniLM, 384 | 518,300 | 796 MB | 15.1 | 59.6 (p95 73.6) | 23.6 | 6.0 |
| bge-base, 768 | 5,183 | 16 MB | 0.03 | 0.38 (p95 2.3) | 0.26 | 0.10 |
| bge-base, 768 | 51,830 | 159 MB | 2.9 | 7.2 (p95 11.4) | 3.3 | 0.85 |
| bge-base, 768 | 518,300 | 1,592 MB | 31.2 | 73.4 (p95 87.4) | 39.4 | 7.8 |

Encoding one query takes a median 6.9 ms with MiniLM and 33.5 ms with bge-base.

- Exact search is cheap at the size of SciFact: under 0.4 ms per query including the sort, a hundredth of the query encoding time. At 518,300 vectors it takes 24 to 39 ms with a partial sort, comparable to encoding the query (bge-base: 39 ms plus 34 ms).
- The matrix-vector product is memory-bound at large size: 796 MB in 15.1 ms and 1,592 MB in 31.2 ms are both about 51 to 53 GB/s, so time grows with the bytes read (dimension times number of vectors). bge-base costs about twice what MiniLM costs at 518,300 vectors. At 51,830 vectors the gap is larger (0.19 against 2.9 ms), probably because the 80 MB matrix is cached and the 159 MB one is not; this was not tested.
- The current implementation sorts all scores. That costs 42 to 44 ms more than the product at 518,300 vectors: three times the product for MiniLM, 1.4 times for bge-base. Taking the top 100 with a partition first cuts it to 8 ms more, and the two give the same top 100 on the SciFact corpus (checked on 20 queries; with copies the ranking has ties, so only the timing is meaningful).
- Scoring the 165 queries as one matrix product is 2.5 to 4 times cheaper per query than one by one (6.0 and 7.8 ms at 518,300 vectors), which matters for offline evaluation and not for a single interactive query.
- Percentiles at the smallest size are noisy (p95 up to 4.6 ms against a median of 0.26 ms); only the larger sizes are stable.
- Exact search at 500,000 vectors is already tolerable on 16 cores. The approximate-index experiment has to be judged against 24 to 39 ms per query and 0.8 to 1.6 GB of memory, not against the 0.4 ms at SciFact size.

## Approximate indexes against exact search (`src/bench_ann.py`)

bge-base embeddings (768 numbers), FAISS 1.15.1 on the CPU, inner product on normalised vectors, one query at a time, top 10, the 165 validation queries. Two indexes besides the exact FAISS flat index: IVF (inverted file: vectors grouped in `nlist` cells by k-means, the query scans the `nprobe` nearest cells) and HNSW (a navigable graph, M = 16, efConstruction = 100, `efSearch` sets how many candidates are explored). Neither compresses the vectors.

On the real SciFact corpus (5,183 vectors), quality is measured against the relevance judgements as well as against the exact top 10:

| index | setting | median ms | overlap with exact top 10 | nDCG@10 | recall@10 |
|---|---|---|---|---|---|
| flat (exact) | n/a | 0.212 | 1.000 | 0.759 | 0.894 |
| IVF, 64 cells | nprobe 1 | 0.015 | 0.444 | 0.473 | 0.528 |
| IVF, 64 cells | nprobe 4 | 0.027 | 0.742 | 0.685 | 0.787 |
| IVF, 64 cells | nprobe 16 | 0.072 | 0.945 | 0.755 | 0.888 |
| HNSW | efSearch 16 | 0.030 | 0.905 | 0.696 | 0.815 |
| HNSW | efSearch 32 | 0.038 | 0.967 | 0.747 | 0.876 |
| HNSW | efSearch 64 | 0.061 | 0.990 | 0.760 | 0.894 |
| HNSW | efSearch 256 | 0.185 | 1.000 | 0.759 | 0.894 |

For speed at scale, the embeddings are replaced by a synthetic corpus of 518,300 vectors: the 5,183 real vectors plus 99 noisy copies of each (Gaussian noise of 0.02 per dimension, renormalised; mean cosine between a copy and its original 0.87). A copy of a document is nearly tied with it, so two measures are given: overlap of the exact top 10 by vector id, and overlap by source document (id modulo 5,183), which ignores which copy was returned. Median ms per query; the exact flat search took 42 to 53 ms in two runs.

| index | setting | median ms | p95 ms | id overlap | source-document overlap | mean score of top 10 against exact |
|---|---|---|---|---|---|---|
| IVF, 2,048 cells | nprobe 1 | 0.16 | 0.35 | 0.561 | 0.505 | 0.975 |
| IVF, 2,048 cells | nprobe 4 | 0.39 | 0.68 | 0.789 | 0.767 | 0.991 |
| IVF, 2,048 cells | nprobe 16 | 1.21 | 1.86 | 0.938 | 0.939 | 0.998 |
| IVF, 2,048 cells | nprobe 64 | 4.05 | 5.12 | 0.988 | 0.993 | 1.000 |
| HNSW | efSearch 64 | 0.18 | 0.35 | 0.575 | 0.620 | 0.978 |
| HNSW | efSearch 256 | 0.54 | 0.86 | 0.772 | 0.868 | 0.993 |
| HNSW | efSearch 512 | 0.99 | 1.41 | 0.833 | 0.938 | 0.997 |
| HNSW | efSearch 1,024 | 1.85 | 2.46 | 0.874 | 0.981 | 0.999 |

Build time: IVF 12.5 s (including training on 100,000 vectors), HNSW 33 s, flat 1.3 s; at 5,183 vectors 0.09 s, 0.16 s and 0.03 s.

- At the size of SciFact an approximate index buys nothing worth having: exact search takes 0.21 ms, and HNSW at efSearch 64 reaches the same nDCG@10 (0.760) in 0.06 ms. The 0.15 ms saved is far below the query encoding time (34 ms for bge-base). The accuracy cost is real if the setting is too small: IVF with one probe falls to 0.473 nDCG@10.
- At 518,300 vectors approximate search is about 35 to 50 times faster than exact search at roughly 0.94 overlap: IVF with 16 probes 1.2 ms, HNSW with efSearch 512 1.0 ms, against 42 to 53 ms exact. For 0.98 to 0.99 overlap IVF needs 4 ms (64 probes, 10 to 13 times faster than exact) and HNSW 1.8 ms (efSearch 1,024, 23 to 28 times faster).
- Speed against accuracy is a curve, not a point: each further halving of the missing neighbours costs roughly two to three times the time, for both indexes. HNSW with M = 16 needs an efSearch of 512 or more before it passes 0.93 on this corpus, much higher than on SciFact itself, where 64 was enough.
- The returned scores stay close to the exact ones even when the ids differ (score ratio 0.975 to 0.998 for most settings): the missed neighbours are mostly near-ties, which suggests the id overlap overstates the damage on this corpus.
- Limits: the large corpus is synthetic, made of tight clusters of 100 near-duplicates, which is harder for a graph index than for the real distribution and unlike real documents; the absolute recall figures at scale do not transfer, and the timings are the more reliable result. Index memory was not measured (both indexes store the full 1.6 GB of vectors; HNSW adds its graph), quantisation (PQ) was not tried, one build per index, timings from one run each on 16 logical cores of a WSL2 machine.

## Cross-encoder re-ranking of the hybrid's top candidates (validation, `src/rerank.py`)

The first stage is the hybrid (BM25 + bge-base, alpha 0.2), nDCG@10 0.794, recall@10 0.908. The cross-encoder scores (query, title + abstract) pairs for the first 50 results and reorders them; results below the re-ranked part keep their first-stage order, so recall@100 is unchanged. `cross-encoder/ms-marco-MiniLM-L6-v2` (22M parameters, trained on MS MARCO web passages, not fine-tuned here), maximum 512 tokens, CPU, 165 validation queries. The ceiling for a perfect re-ranker over these candidates is 0.934 (20 candidates) and 0.950 (50).

| candidates re-ranked | nDCG@10 | MRR@10 | recall@10 | difference from the hybrid | better / worse |
|---|---|---|---|---|---|
| none (hybrid) | 0.794 (0.742 to 0.842) | 0.764 | 0.908 | n/a | n/a |
| top 20 | 0.745 (0.691 to 0.797) | 0.706 | 0.886 | -0.048 (-0.077 to -0.020) | 12 / 40 |
| top 50 | 0.737 (0.682 to 0.790) | 0.698 | 0.878 | -0.057 (-0.088 to -0.027) | 11 / 41 |

Cost per query on 16 logical cores: re-scoring 50 candidates takes a median 1.77 s (p95 2.15 s) with batches of 16 and 2.68 s with one batch of 50 (padding to 512 tokens makes the large batch slower). The whole first stage (BM25, bge-base query encoding, scoring all 5,183 documents) takes a median 104 ms, so the re-ranker costs about 17 times the first stage. `BAAI/bge-reranker-base` (278M parameters) takes a median 10.6 s per query (p95 12.1 s) for 50 candidates with batches of 16, and 13.5 s with one batch, measured on 5 queries; its quality was not evaluated, because a full validation run takes about 30 minutes and was stopped.

- The MS MARCO MiniLM cross-encoder makes the ranking worse: -0.057 nDCG@10 over 50 candidates (interval below zero), worse on 41 queries and better on 11. Re-ranking fewer candidates hurts less (-0.048 for 20) but still hurts, so the damage is in the model's ordering of the top results, not in how far down it looks. A likely cause is the mismatch between web-passage relevance and SciFact's task of finding the paper that supports a claim; this was not tested, and the model was not fine-tuned on SciFact.
- Even a re-ranker that worked would be expensive: the ceiling of 0.950 at 50 candidates is 0.156 above the hybrid, but each query costs 1.8 s with the small model and about 11 s with the base model, against 0.1 s for the first stage.
- The re-ranker was not scored on the test queries: it is a negative result on validation, not a final system.

## Not done yet

API and page.
