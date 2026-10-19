import httpx
import pytest

from harness.corpus import openalex as oa


def test_inverted_index_repeated_words_and_gaps():
    idx = {"the": [0, 3], "cat": [1], "sat": [2], "mat": [4]}
    assert oa.abstract_from_inverted_index(idx) == "the cat sat the mat"
    assert oa.abstract_from_inverted_index({"a": [0], "b": [5]}) == "a b"  # gaps just close up
    assert oa.abstract_from_inverted_index(None) is None
    assert oa.abstract_from_inverted_index({}) is None


def _work(**kw):
    w = {"id": "https://openalex.org/W123", "doi": "https://doi.org/10.1/abc", "title": "Deep learning in schools",
         "publication_year": 2020, "cited_by_count": 42,
         "authorships": [{"author": {"display_name": "Jane Q. Smith"}}],
         "primary_location": {"source": {"display_name": "Journal of Testing"}},
         "biblio": {"volume": "12", "issue": "3", "first_page": "1", "last_page": "9"},
         "abstract_inverted_index": {"We": [0], "study": [1], "schools": [2]}}
    w.update(kw)
    return w


def test_work_to_ref_one_two_three_authors():
    r = oa.work_to_ref(_work())
    assert r["openalex_id"] == "W123" and r["doi"] == "10.1/abc" and r["pages"] == "1-9"
    assert r["apa"] == "Smith, J. Q. (2020). Deep learning in schools. Journal of Testing, 12(3), 1-9. https://doi.org/10.1/abc"
    assert r["abstract"] == "We study schools" and r["cited_by"] == 42

    two = oa.work_to_ref(_work(authorships=[{"author": {"display_name": "Jane Smith"}}, {"author": {"display_name": "Li Wei Chen"}}]))
    assert two["authors"] == "Smith, J., and Chen, L. W."
    assert "Smith, J., & Chen, L. W. (2020)." in two["apa"]

    three = oa.work_to_ref(_work(authorships=[{"author": {"display_name": n}} for n in ("A Lee", "B Kim", "C Park")]))
    assert "Lee, A., Kim, B., & Park, C. (2020)." in three["apa"]


def test_work_to_ref_missing_pages_and_drops_bad_rows():
    r = oa.work_to_ref(_work(biblio={"volume": "1"}))
    assert r["pages"] is None and "Journal of Testing, 1." in r["apa"]
    assert oa.work_to_ref(_work(authorships=[])) is None
    assert oa.work_to_ref(_work(title="")) is None
    assert oa.work_to_ref(_work(publication_year=None)) is None


def test_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "")
    assert oa.enabled() is False
    with pytest.raises(oa.OpenAlexError, match="OPENALEX_API_KEY"):
        oa.get("/works")


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_429_then_200_retries(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "k")
    monkeypatch.setattr(oa.time, "sleep", lambda s: None)
    calls = []

    def handler(req):
        calls.append(req.url.params)
        return httpx.Response(429) if len(calls) == 1 else httpx.Response(200, json={"ok": True})

    assert oa.get("/works", {"x": "1"}, client=_client(handler)) == {"ok": True}
    assert len(calls) == 2 and calls[0]["api_key"] == "k" and calls[0]["mailto"] == oa.MAILTO


def test_persistent_500_and_4xx_raise(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "k")
    monkeypatch.setattr(oa.time, "sleep", lambda s: None)
    with pytest.raises(oa.OpenAlexError) as e:
        oa.get("/works", client=_client(lambda r: httpx.Response(503)))
    assert e.value.status == 503
    with pytest.raises(oa.OpenAlexError) as e:
        oa.get("/works", client=_client(lambda r: httpx.Response(403, text="nope")))
    assert e.value.status == 403


def test_search_works_filters_and_drops_retracted(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "k")
    seen = {}

    def handler(req):
        seen.update(dict(req.url.params))
        return httpx.Response(200, json={"results": [
            _work(), _work(id="https://openalex.org/W9", is_retracted=True), _work(id="https://openalex.org/W8", authorships=[])]})

    refs = oa.search_works("deep learning", from_year=2015, limit=8, client=_client(handler))
    assert [r["openalex_id"] for r in refs] == ["W123"]
    assert seen["filter"] == "type:article|review,is_retracted:false,has_abstract:true,from_publication_date:2015-01-01"
    assert seen["per-page"] == "8" and seen["search"] == "deep learning"


def test_find_cited_work_needs_the_surname_AND_topic_words_to_match(monkeypatch):
    from harness.corpus import openalex as oa

    def work(wid, first, title, abstract=None):
        return {"id": f"https://openalex.org/{wid}", "title": title, "publication_year": 1990, "cited_by_count": 5,
                "authorships": [{"author": {"display_name": first}}],
                "abstract_inverted_index": {w: [i] for i, w in enumerate(abstract.split())} if abstract else None}
    calls = []
    hint = "Thesis on employee engagement and burnout. Personal engagement varies with psychological conditions at work"

    def fake_get(path, params, client=None):
        calls.append(params)
        return {"results": [
            work("W1", "Ann Kahne", "Employee engagement and burnout at work"),                # wrong surname
            work("W2", "William Kahn", "Structure of human enzyme transferase", "protein folding kinetics"),   # namesake, other topic
            work("W3", "William Kahn", "Psychological conditions of personal engagement at work")]}         # right one
    monkeypatch.setattr(oa, "get", fake_get)
    assert oa.find_cited_work("Kahn", 1990, hint)["openalex_id"] == "W3"
    assert "raw_author_name.search:Kahn" in calls[0]["filter"] and "publication_year:1990" in calls[0]["filter"]
    monkeypatch.setattr(oa, "get", lambda p, q, client=None: {"results": [work("W2", "William Kahn", "Enzyme transferase folding")]})
    assert oa.find_cited_work("Kahn", 1990, hint) is None                 # same surname and year, unrelated topic
    assert oa.find_cited_work("Kahn", 1990, "short") is None and oa.find_cited_work("", 2015, hint) is None


def test_find_cited_work_rejects_a_namesake_whose_title_shares_no_topic_word(monkeypatch):
    from harness.corpus import openalex as oa
    hint = "Structural health monitoring of bridges using wireless sensor networks and damage detection"
    other = {"id": "https://openalex.org/W5", "title": "Inhaled therapy for lung disease", "publication_year": 2017,
             "cited_by_count": 50, "authorships": [{"author": {"display_name": "Anna Sabato"}}],
             # abstract shares topic words, but the title shares none: still a namesake, not the cited work
             "abstract_inverted_index": {"monitoring": [0], "structural": [1], "networks": [2], "patients": [3]}}
    monkeypatch.setattr(oa, "get", lambda p, q, client=None: {"results": [other]})
    assert oa.find_cited_work("Sabato", 2017, hint) is None
