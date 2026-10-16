from src.baselines import doc_words, random_system, shortest_first_system

CORPUS = {
    "a": {"title": "one two", "text": "three four five"},
    "b": {"title": "one", "text": "two"},
    "c": {"title": "one", "text": "two three"},
    "d": {"title": "x", "text": "y"},
}


def test_shortest_first_orders_by_word_count_then_id():
    ranking = shortest_first_system(CORPUS)("any query")
    assert ranking == ["b", "d", "c", "a"]
    assert [doc_words(CORPUS[d]) for d in ranking] == [2, 2, 3, 5]


def test_shortest_first_ignores_the_query():
    system = shortest_first_system(CORPUS)
    assert system("one query") == system("another query")


def test_random_system_is_reproducible_per_seed():
    ids = [f"d{i}" for i in range(500)]
    assert random_system(ids, 1)("q") == random_system(ids, 1)("q")
    assert random_system(ids, 1)("q") != random_system(ids, 2)("q")


def test_random_system_returns_distinct_documents_and_varies_per_query():
    ids = [f"d{i}" for i in range(500)]
    system = random_system(ids, 0)
    first, second = system("q"), system("q")
    assert len(set(first)) == len(first) == 100
    assert first != second
