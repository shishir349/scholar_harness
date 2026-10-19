from harness.text.apa import build_apa
from harness.text.similarity import title_similarity
from harness.text.sentences import split_sentences, words, paragraphs
from harness.text.citations import extract_citations, audit_citations
from harness.text.strip import strip_for_style, headings_and_text

ARTICLE = dict(authors="Bandura, A.", year=1977,
               title="Self-efficacy: Toward a unifying theory of behavioral change",
               journal="Psychological Review", volume="84", issue="2", pages="191–215", type="JOURNAL")


# ── apa.js parity (cases from apa.test.js) ──────────────────────────────────
def test_apa_journal_article():
    assert build_apa(ARTICLE) == ("Bandura, A. (1977). Self-efficacy: Toward a unifying theory of behavioral change. "
                                  "Psychological Review, 84(2), 191–215.")


def test_apa_no_markdown_italics():
    assert "*" not in build_apa(ARTICLE)
    assert "*" not in build_apa({**ARTICLE, "type": "BOOK"})


def test_apa_book_publisher_slot():
    assert build_apa(dict(authors="Senge, P. M.", year=1990, title="The fifth discipline",
                          journal="Doubleday", type="BOOK")) == "Senge, P. M. (1990). The fifth discipline. Doubleday."


def test_apa_no_year_and_doi_url_rules():
    assert "(n.d.)" in build_apa(dict(authors="Smith, J.", title="Untitled work"))
    assert build_apa({**ARTICLE, "doi": "10.1/x", "url": "http://e.com"}).endswith("https://doi.org/10.1/x")
    assert build_apa({**ARTICLE, "doi": "https://doi.org/10.1/x"}).endswith("doi.org/10.1/x")
    assert build_apa({**ARTICLE, "url": "http://e.com"}).endswith("http://e.com")


def test_apa_no_double_period_and_ampersand_and_missing_pieces():
    assert ".." not in build_apa({**ARTICLE, "title": "Ends already."})
    assert "Smith, J., & Jones, M." in build_apa({**ARTICLE, "authors": "Smith, J. and Jones, M."})
    assert build_apa(dict(authors="Lee, A.", year=2020, title="A paper", type="JOURNAL")) == "Lee, A. (2020). A paper."
    assert build_apa(dict(authors="Lee, A.", year=2020, title="T", pages="1–5")).endswith("1–5.")


# ── similarity ───────────────────────────────────────────────────────────────
def test_title_similarity():
    assert title_similarity("Deep learning for medical imaging", "Medical imaging with deep learning") == 1.0
    assert title_similarity("Art of it", "Art") is None  # words <= 3 chars ignored: no opinion, never a mismatch
    assert title_similarity("Quantum gravity loops", "Cooking pasta tonight") == 0.0


# ── sentences ────────────────────────────────────────────────────────────────
def test_split_sentences_protects_abbreviations_and_decimals():
    s = split_sentences("Smith et al. (2019) found r = 0.45, e.g. in schools. The effect was small. Why? Unclear.")
    assert s == ["Smith et al. (2019) found r = 0.45, e.g. in schools.", "The effect was small.", "Why?", "Unclear."]


def test_words_and_paragraphs():
    assert len(words("It's a well-known result, 2019.")) == 5
    assert paragraphs("a\n\nb\n\n\nc") == ["a", "b", "c"]


# ── citations (citationExtractor.js behaviour) ──────────────────────────────
def test_extract_citations_forms():
    text = "Prior work (Smith, 2019) and (Smith & Jones, 2020) and (Lee et al., 2018). Brown (2021) agreed."
    got = {(c["author"], c["year"]) for c in extract_citations(text)}
    assert ("Smith", 2019) in got and ("Smith & Jones", 2020) in got
    assert ("Lee et al.", 2018) in got and ("Brown", 2021) in got


def test_narrative_not_double_counted_inside_parentheses():
    assert len(extract_citations("as shown (Smith, 2019).")) == 1


def test_audit_matches_flags_future_and_unmatched():
    refs = [{"authors": "Smith, J.", "year": 2019}, {"authors": "Unused, U.", "year": 2001}]
    cits = extract_citations("(Smith, 2019) (Ghost, 2015) (Future, 2999) (Ancient, 1800)")
    a = audit_citations(cits, refs)
    assert [c["author"] for c in a["matched"]] == ["Smith"]
    assert [c["author"] for c in a["unmatched"]] == ["Ghost"]
    assert {c["author"] for c in a["suspicious"]} == {"Future", "Ancient"}
    assert [r["authors"] for r in a["unused_refs"]] == ["Unused, U."]


# ── strip ────────────────────────────────────────────────────────────────────
def test_strip_removes_figures_tables_headings_and_tags():
    raw = ("<h2>2.1 Title</h2><p>Real prose here.</p>[FIGURE_START]\nTYPE: mermaid\n[FIGURE_END]"
           "\n| a | b |\n|---|---|\n<p>More prose &amp; more.</p>")
    out = strip_for_style(raw)
    assert "FIGURE" not in out and "2.1 Title" not in out and "|" not in out
    assert "Real prose here." in out and "More prose & more." in out
    assert "# 2.1 Title" in headings_and_text(raw)


def test_first_author_splits_narrative_and_but_not_names_containing_and():
    from harness.text.citations import first_author, match_ref
    assert first_author("Macey and Schneider") == "Macey" and first_author("Smith & Lee") == "Smith"
    assert first_author("Anderson") == "Anderson" and first_author("Sandberg and Lee") == "Sandberg"
    assert match_ref({"author": "Macey and Schneider", "year": 2008},
                     [{"year": 2008, "authors": "Macey, W. H., & Schneider, B."}])
