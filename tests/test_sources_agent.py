import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from harness import llm
from harness.sources import agent as A
from harness.sources.tools import NOTE_CHARS, Ledger, merge_refs, source_notes_for, compact


class Rec(GenericFakeChatModel):
    seen: list = []

    def _generate(self, messages, *a, **k):
        self.seen.append(messages[-1].content)
        return super()._generate(messages, *a, **k)


@pytest.fixture(autouse=True)
def restore():
    yield
    llm.set_fake(None)


def ref(oa, title, cited=10, doi=None, year=2020, authors="Smith, J., & Lee, K."):
    return {"openalex_id": oa, "title": title, "year": year, "authors": authors, "doi": doi, "journal": "J",
            "cited_by": cited, "abstract": f"Abstract of {title}. " * 5, "apa": f"{authors} ({year}). {title}."}


def fake_search(results_by_query):
    def fn(q, y, n):
        return results_by_query.get(q, [])
    return fn


THESIS = {"title": "T", "domain": "D", "methodology": "M", "toc_data": {"objectives": ["o1"], "questions": ["q1"]}}
CHAPTER = {"chapter_number": 2, "title": "Literature Review"}
SECTIONS = [{"number": "2.1", "title": "A"}, {"number": "2.2", "title": "B"}]
PLAN = json.dumps({"queries": [{"q": "q1"}, {"q": "q2"}]})
FINAL = json.dumps({"selected": [{"id": "S1", "sections": ["2.1"]}, {"id": "S2", "sections": ["2.2", "2.1"]},
                                  {"id": "S99", "sections": ["2.1"]}]})


def run(replies, search, **kw):
    """Only the haiku tier ever gets a scripted reply; a sonnet call would starve on an empty iterator and error."""
    haiku = Rec(messages=iter([AIMessage(content=t) for t in replies]))
    sonnet = Rec(messages=iter([]))
    llm.set_fake(lambda tier: haiku if tier == "haiku" else sonnet)
    return A.research_chapter(THESIS, CHAPTER, SECTIONS, search_fn=fake_search(search), **kw), haiku, sonnet


def test_two_calls_total_plan_then_select_on_the_cheap_tier():
    r, haiku, sonnet = run([PLAN, FINAL], {"q1": [ref("W1", "One", 5)], "q2": [ref("W2", "Two", 50)]})
    assert [s["ledger_id"] for s in r["sources"]] == ["S1", "S2"]          # S99 unknown -> dropped
    assert r["by_section"] == {"2.1": ["S1", "S2"], "2.2": ["S2"]}
    assert r["searches"] == 2 and r["rounds"] == 2
    assert len(haiku.seen) == 2 and len(sonnet.seen) == 0                 # sonnet never touched


def test_plan_parse_failure_yields_no_sources_without_crashing():
    r, *_ = run(["not json", FINAL], {"q1": [ref("W1", "One")]})
    assert r == {"sources": [], "by_section": {}, "rounds": 2, "searches": 0}


def test_select_parse_failure_falls_back_to_top_cited():
    r, *_ = run([PLAN, "I could not format that"], {"q1": [ref("W1", "One", 5), ref("W2", "Two", 50)]})
    assert [s["openalex_id"] for s in r["sources"]] == ["W2", "W1"] and r["by_section"] == {}


def test_one_bad_query_does_not_sink_the_others():
    def boom_then_ok(q, y, n):
        if q == "q1":
            raise RuntimeError("down")
        return [ref("W2", "Two")]
    haiku = Rec(messages=iter([AIMessage(content=PLAN), AIMessage(content=FINAL)]))
    llm.set_fake(lambda tier: haiku if tier == "haiku" else Rec(messages=iter([])))
    r = A.research_chapter(THESIS, CHAPTER, SECTIONS, search_fn=boom_then_ok)
    assert r is not None and r["searches"] == 1


def test_any_exception_returns_none_and_no_key_skips(monkeypatch):
    class Boom(GenericFakeChatModel):
        def _generate(self, *a, **k):
            raise RuntimeError("model down")
    llm.set_fake(lambda tier: Boom(messages=iter([])))
    assert A.research_chapter(THESIS, CHAPTER, SECTIONS, search_fn=fake_search({})) is None
    llm.set_fake(None)
    monkeypatch.setenv("OPENALEX_API_KEY", "")
    assert A.research_chapter(THESIS, CHAPTER, SECTIONS) == {"sources": [], "by_section": {}, "skipped": "no_key"}


def test_budget_exceeded_is_not_swallowed():
    from harness.spend import BudgetExceeded

    class Over(GenericFakeChatModel):
        def _generate(self, *a, **k):
            raise BudgetExceeded("over")
    llm.set_fake(lambda tier: Over(messages=iter([])))
    with pytest.raises(BudgetExceeded):
        A.research_chapter(THESIS, CHAPTER, SECTIONS, search_fn=fake_search({}))


def test_merge_refs_dedupes_by_doi_id_and_title():
    lib = [{"title": "Deep learning for schools", "doi": "10.1/A", "authors": "X"}, {"title": "Other", "openalex_id": "W7"}]
    src = [ref("W1", "Fresh paper", doi="10.1/new"), ref("W2", "Whatever", doi="10.1/a"), ref("W7", "Renamed"),
           ref("W3", "Deep learning for schools today"), ref("W4", "Totally different topic here")]
    out = merge_refs(lib, src)
    assert [r["title"] for r in out] == ["Deep learning for schools", "Other", "Fresh paper", "Totally different topic here"]


def test_source_notes_mapped_fallback_and_empty():
    s1, s2 = ref("W1", "One", 5), ref("W2", "Two", 50)
    l = Ledger()
    l.add(s1), l.add(s2)
    srcs = l.items
    mapped = source_notes_for({"number": "2.1"}, srcs, {"2.1": ["S1"]})
    assert "(Smith, 2020)" in mapped and "Abstract of One" in mapped and "Abstract of Two" not in mapped
    fallback = source_notes_for({"number": "9.9"}, srcs, {"2.1": ["S1"]})
    assert "Abstract of Two" in fallback
    assert source_notes_for({"number": "1"}, [], None) == ""
    assert compact(srcs[0])["id"] == "S1"


def test_planner_asks_for_exact_foundational_titles_and_selector_keeps_them():
    assert "EXACT titles" in A.PLAN_SYSTEM and "never invent a title" in A.PLAN_SYSTEM
    assert "foundational" in A.SELECT_SYSTEM



def test_source_notes_keep_abstracts_short_to_hold_the_prompt_small():
    srcs = [{"ledger_id": "S1", "authors": "Lee, K.", "year": 2020, "abstract": "x" * 600, "cited_by": 3}]
    line = source_notes_for({"number": "2.1"}, srcs, {"2.1": ["S1"]}).splitlines()[-1]
    assert line.count("x") == NOTE_CHARS


def test_research_defaults_to_a_short_reference_list():
    import inspect
    from harness.sources.agent import research_chapter
    assert inspect.signature(research_chapter).parameters["max_sources"].default == 12


def test_source_notes_keep_the_sentence_that_carries_the_figures():
    abstract = ("This study looks at how low cost wireless accelerometer nodes behave when mounted on a long span cable stayed bridge "
                "over several seasons and under many different traffic loads that were recorded over a very long monitoring campaign by several research teams working together across different countries. Accuracy reached 94% across 120 samples at the "
                "Pietratagliata bridge. Further work is still needed.")
    srcs = [{"ledger_id": "S1", "authors": "Lee, K.", "year": 2020, "abstract": abstract, "cited_by": 3}]
    note = source_notes_for({"number": "2.1"}, srcs, {"2.1": ["S1"]}).splitlines()[-1].split("): ", 1)[1]
    assert "94%" in note and "120 samples" in note and len(note) <= NOTE_CHARS and note.startswith("This study")


def test_source_notes_prefer_a_sentence_with_names_when_no_sentence_has_figures():
    abstract = ("Wireless nodes are widely discussed in the monitoring literature for their low cost and easy installation on old "
                "structures of many kinds, as reported by several research groups over a long period of careful and repeated observation in the field. "
                "The Raspberry Pi board and the ADXL355 sensor were compared on a masonry tower. Results vary.")
    srcs = [{"ledger_id": "S1", "authors": "Lee, K.", "year": 2020, "abstract": abstract, "cited_by": 3}]
    note = source_notes_for({"number": "2.1"}, srcs, {"2.1": ["S1"]}).splitlines()[-1].split("): ", 1)[1]
    assert "ADXL355" in note and len(note) <= NOTE_CHARS
