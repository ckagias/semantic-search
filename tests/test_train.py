from src.train import make_batches


def pair(qid, doc_id):
    return {"qid": qid, "query": f"q{qid}", "doc_id": doc_id, "doc": f"d{doc_id}"}


PAIRS = [pair(str(i), str(i % 7)) for i in range(40)] + [pair("0", "99"), pair("1", "99")]


def test_no_duplicate_query_or_document_inside_a_batch():
    for batch in make_batches(PAIRS, batch_size=8, seed=0):
        assert len({p["qid"] for p in batch}) == len(batch)
        assert len({p["doc_id"] for p in batch}) == len(batch)


def test_batches_respect_size_and_cover_each_pair_at_most_once():
    batches = make_batches(PAIRS, batch_size=5, seed=1)
    assert all(2 <= len(b) <= 5 for b in batches)
    used = [(p["qid"], p["doc_id"]) for b in batches for p in b]
    assert len(used) == len(set(used))
    assert set(used) <= {(p["qid"], p["doc_id"]) for p in PAIRS}


def test_batches_depend_on_the_seed_only():
    first = make_batches(PAIRS, batch_size=8, seed=3)
    assert first == make_batches(PAIRS, batch_size=8, seed=3)
    assert first != make_batches(PAIRS, batch_size=8, seed=4)


def test_a_document_with_many_pairs_lands_in_separate_batches():
    pairs = [pair(str(i), "same") for i in range(4)]
    batches = make_batches(pairs + [pair("x", "other"), pair("y", "other2")], batch_size=4, seed=0)
    assert all(sum(p["doc_id"] == "same" for p in b) <= 1 for b in batches)
