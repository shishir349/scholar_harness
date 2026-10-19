from harness.validate.overlap import overlap_report
from harness.validate.structure import check_structure


def ids(res, want=True):
    return {c["id"] for c in res["checks"] if c["passed"] == want}


THESIS = {"methodology": "Quantitative survey", "questions": ["How does digital adoption affect firm performance?"]}


def chap(secs=None, target=None):
    return {"toc_sections": {"sections": secs or []}, "target_words": target}


# ── overlap ──────────────────────────────────────────────────────────────────
EX = "the quick brown fox jumps over the lazy dog while the cat watches quietly from the old wooden fence nearby"


def test_verbatim_copy_flagged_paraphrase_not_empty_not():
    copy = "Intro words here. " + " ".join(EX.split()[:15]) + " and then more of our own text follows on."
    r = overlap_report(copy, [EX])
    assert r["flagged"] and r["max_run"] >= 12
    para = "A fast auburn fox leapt across a sleepy hound as a feline observed silently from a weathered timber railing."
    assert not overlap_report(para, [EX])["flagged"]
    assert overlap_report(copy, [])["flagged"] is False
    assert overlap_report("short", [EX]) == {"max_run": 0, "shared_ratio": 0.0, "flagged": False}


def test_short_shared_run_below_threshold_not_flagged():
    d = ("completely different opening sentence about markets and firms. the quick brown fox jumps over the lazy dog "
         + "unrelated words " * 60)
    r = overlap_report(d, [EX])
    assert r["max_run"] < 12 and not r["flagged"]


# ── structure ────────────────────────────────────────────────────────────────
def test_introduction_pass_and_fail():
    good = ("<h2>1.1 Background</h2>This research question and objective address a gap: little is known about firms. "
            "Chapter two reviews the literature. " * 3)
    r = check_structure(good, chap([{"number": "1.1", "title": "Background"}]), THESIS, "introduction")
    assert r["pass_rate"] == 1.0
    bad = check_structure("Plain prose about nothing in particular. " * 20, chap([{"number": "1.1", "title": "Background"}]), THESIS, "introduction")
    assert {"states_rq_or_objectives", "gap_stated", "chapter_preview", "section_present:1.1"} <= ids(bad, False)


def test_literature_review():
    cites = " ".join(f"(Author{chr(65 + i)}, 20{10 + i})" for i in range(6))
    body = ("Prior work is broad " + cites + ". ") + ("filler words sit here plainly. " * 60) + "Yet a gap remains unaddressed in this area."
    r = check_structure(body, chap(), THESIS, "literature_review")
    assert ids(r) == {"gap_near_end", "distinct_sources"}
    r2 = check_structure("Gap early. " + "filler words sit here plainly. " * 60, chap(), THESIS, "literature_review")
    assert {"gap_near_end", "distinct_sources"} <= ids(r2, False)


def test_methodology_groups():
    full = ("Our research design follows a positivist paradigm. The sample comprised 200 participants. Data collection used a "
            "questionnaire. Analysis used regression. Ethical approval was granted. Validity and reliability were checked.")
    assert check_structure(full, chap(), THESIS, "methodology")["pass_rate"] == 1.0
    partial = check_structure("The sample comprised participants.", chap(), THESIS, "methodology")
    assert "covers:ethics" in ids(partial, False) and "covers:sample/participants" in ids(partial)


def test_findings_numbers_only_for_quantitative():
    nums = "Mean 4.2, SD 1.1, n = 200, p < 0.05. " * 10
    assert check_structure(nums, chap(), THESIS, "findings")["pass_rate"] == 1.0
    assert not check_structure("words only here " * 40, chap(), THESIS, "findings")["pass_rate"]
    qual = check_structure("words only here " * 40, chap(), {"methodology": "Qualitative"}, "findings")
    assert qual["checks"] == [] and qual["pass_rate"] == 1.0


def test_discussion_and_rq_coverage():
    cites = " ".join(f"(Author{chr(65 + i)}, 20{10 + i})" for i in range(5))
    good = f"The findings show digital adoption affects firm performance {cites}. " * 2
    r = check_structure(good, chap(), THESIS, "discussion")
    assert ids(r, False) == set()
    bad = check_structure("The results are interesting and weather is nice.", chap(), THESIS, "discussion")
    assert {"distinct_citations", "answers_rq1"} <= ids(bad, False)


def test_conclusion_mentions_and_word_count():
    txt = "Our contribution and implications are clear; future research and limitations remain. Digital adoption affects firm performance."
    r = check_structure(txt, chap(target=100), THESIS, "conclusion")
    assert "word_count" in ids(r, False) and {"mentions:contribution", "mentions:limitation", "answers_rq1"} <= ids(r)
    assert check_structure("nothing relevant", chap(), THESIS, "conclusion")["pass_rate"] < 0.3
