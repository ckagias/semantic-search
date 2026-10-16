import statistics as st
from collections import Counter

from src.data import load_corpus, load_qrels, load_queries


def words(doc):
    return len((doc["title"] + " " + doc["text"]).split())


def pct(values, p):
    values = sorted(values)
    return values[int(p * (len(values) - 1))]


def main():
    corpus = load_corpus()
    queries = load_queries()
    qrels = {"train": load_qrels("train"), "test": load_qrels("test")}

    lengths = [words(d) for d in corpus.values()]
    print(
        f"documents: {len(corpus)}, words mean {st.mean(lengths):.0f}, "
        f"median {st.median(lengths):.0f}, p5 {pct(lengths, .05)}, "
        f"p95 {pct(lengths, .95)}, max {max(lengths)}"
    )
    judged = [qid for r in qrels.values() for qid in r]
    qlen = [len(queries[qid].split()) for qid in judged]
    print(f"queries: {len(judged)}, words mean {st.mean(qlen):.1f}, max {max(qlen)}")

    for name, r in qrels.items():
        counts = Counter(len(v) for v in r.values())
        print(f"{name}: {len(r)} queries, relevant per query {sorted(counts.items())}")

    rel = {name: {d for v in r.values() for d in v} for name, r in qrels.items()}
    print(
        f"relevant documents: train {len(rel['train'])}, test {len(rel['test'])}, "
        f"in both {len(rel['train'] & rel['test'])}"
    )
    shared = sum(1 for v in qrels["test"].values() if set(v) & rel["train"])
    print(f"test queries with a relevant document also relevant to a train query: "
          f"{shared} of {len(qrels['test'])}")
    dup = {queries[q] for q in qrels["train"]} & {queries[q] for q in qrels["test"]}
    print(f"query texts present in both splits: {len(dup)}")


if __name__ == "__main__":
    main()
