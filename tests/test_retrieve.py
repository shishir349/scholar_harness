import pytest

from harness.corpus import store, retrieve

FIELDS = [
    {"name": "Computer Science", "keywords": ["machine learning", "neural networks", "software engineering"]},
    {"name": "Business, Management and Accounting", "keywords": ["marketing strategy", "organizational behavior", "supply chain"]},
]


@pytest.fixture
def con():
    retrieve.clear_cache()
    c = store.connect(":memory:")

    def thesis(oid, field, score, holdout=0):
        return store.upsert_thesis(c, dict(openalex_id=oid, title=f"T {oid}", field_id="1", field_name=field,
                                           score=score, holdout=holdout))

    t1 = thesis("W1", "Computer Science", 90)
    t2 = thesis("W2", "Computer Science", 70)
    t3 = thesis("W3", "Business, Management and Accounting", 80)
    th = thesis("W4", "Computer Science", 99, holdout=1)

    def chunk(tid, field, txt, pos="open", st="literature_review", idx=0):
        return dict(section_type=st, position=pos, heading="H", chunk_idx=idx, text=txt)

    store.insert_chunks(c, t1, "Computer Science", [chunk(t1, "", "neural networks learn representations from data", idx=0),
                                                    chunk(t1, "", "second chunk about neural networks training", idx=1)])
    store.insert_chunks(c, t2, "Computer Science", [chunk(t2, "", "software engineering practice in teams")])
    store.insert_chunks(c, t3, "Business, Management and Accounting", [chunk(t3, "", "marketing strategy of firms")])
    store.insert_chunks(c, th, "Computer Science", [chunk(th, "", "neural networks holdout secret text")])
    store.insert_chunks(c, t1, "Computer Science", [chunk(t1, "", "a mid chunk about neural networks", pos="mid", idx=2)])
    c.commit()
    yield c
    c.close()


def _get(con, **kw):
    base = dict(con=con, fields=FIELDS, title="Neural networks for image tasks", domain="Computer Science machine learning",
                chapter_title="Literature Review", section_title="Neural networks", section_type="literature_review",
                position="open", limit=2)
    base.update(kw)
    return retrieve.get_exemplars(**base)


def test_topic_query_is_safe_and_bounded():
    q = retrieve.topic_query('neural " ) OR * NEAR( networks; DROP TABLE users -- learning deep vision data theory model extra')
    assert q.count(" OR ") <= 7 and '"' not in q and "*" not in q and "(" not in q and ";" not in q
    assert "neural" in q and retrieve.topic_query("") == "" and retrieve.topic_query("a an the of") == ""


def test_map_to_field():
    assert retrieve.map_to_field("Deep neural networks", "machine learning", FIELDS) == "Computer Science"
    assert retrieve.map_to_field("zzz qqq", "xxx", FIELDS) is None


def test_field_filter_holdout_and_one_chunk_per_thesis(con):
    got = _get(con)
    assert len(got) == 2
    assert all(g["field"] == "Computer Science" for g in got)
    assert not any("holdout" in g["text"] for g in got)
    ids = {g["id"] for g in got}
    thesis_ids = {con.execute("SELECT thesis_id FROM chunks WHERE id=?", (i,)).fetchone()[0] for i in ids}
    assert len(thesis_ids) == 2  # t1 has two 'open' chunks; only one may be used


def test_position_filter(con):
    got = _get(con, position="mid", limit=3)
    assert [g["text"] for g in got][0].startswith("a mid chunk")


def test_zero_match_query_still_returns_by_score(con):
    got = _get(con, title="zzzz qqqq", domain="", chapter_title="", section_title="")
    assert len(got) == 2  # no keyword match, no field mapped: pooled top-up by thesis score
    assert not any("holdout" in g["text"] for g in got)


def test_thin_field_pools_all_fields(con):
    got = _get(con, limit=3, title="marketing strategy", domain="Business management")
    fields = {g["field"] for g in got}
    assert "Business, Management and Accounting" in fields and len(got) == 3


def test_cache_hit_and_error_returns_empty(con, monkeypatch):
    first = _get(con)
    con.execute("DELETE FROM chunks")
    assert _get(con) == first  # served from cache
    retrieve.clear_cache()
    assert _get(con) == []

    class Boom:
        def execute(self, *a, **k):
            import sqlite3
            raise sqlite3.OperationalError("boom")
    retrieve.clear_cache()
    assert retrieve.get_exemplars(con=Boom(), fields=FIELDS, title="x y z w", domain="", chapter_title="", section_title="",
                                  section_type="literature_review", position="open") == []


def test_build_exemplar_block(con):
    assert retrieve.build_exemplar_block([], "literature_review", "open") == ""
    block = retrieve.build_exemplar_block(_get(con), "literature_review", "open")
    assert block.startswith("STYLE EXEMPLARS:") and "[EXEMPLAR 1: Computer Science, open of a Literature Review section]" in block
    assert "Do NOT copy or paraphrase" in block and block.count('"""') == 4


def test_rerank_prefers_higher_topical_overlap():
    rows = [{"heading": "Unrelated heading", "text": "loosely related content about firms", "id": 1,
             "field_name": "CS", "thesis_id": 1},
            {"heading": "Neural networks representations", "text": "neural networks learn representations from data",
             "id": 2, "field_name": "CS", "thesis_id": 2}]
    ctx = retrieve._tokens("neural networks representations learning")
    assert retrieve._rerank(rows, ctx)[0]["id"] == 2
    assert retrieve._rerank(rows, set()) == rows              # no context: leave bm25 order alone


def test_store_delete_keeps_fts_in_sync(con):
    tid = con.execute("SELECT id FROM theses WHERE openalex_id='W1'").fetchone()[0]
    store.delete_thesis_chunks(con, tid)
    con.commit()
    left = con.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'representations'").fetchone()[0]
    assert left == 0
    assert store.random_holdout_chunk(con, "literature_review", "open")["text"].endswith("secret text")
