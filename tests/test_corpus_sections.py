from harness.corpus.sections import (build_sections, chunk_unit, clean_thesis_text, n_words, split_chapters,
                                     split_units)


def para(n_words_, tag="w"):
    return " ".join(f"{tag}{i}" for i in range(n_words_)) + "."


def thesis_md():
    hdr = "RUNNING HEADER TEXT"
    parts = ["Title page of the thesis", "Acknowledgements", "Thanks to everyone.", "1", ""]
    parts += ["# Chapter 1 Introduction", hdr, para(320), "", para(300), "", "# Chapter 2 Literature Review", hdr,
              "## 2.1 Background", para(200), "", para(200), "", "## 2.2 Tiny", para(50), "",
              "## 2.3 Debates", para(500), "", para(400), "", para(300),
              "# Chapter 3 Research Methodology", hdr, para(500), "",
              "| a | b |", "|---|---|", "Table 3.1 Something", "![img](x.png)", "42",
              "# Chapter 4 Results and Discussion", hdr, para(900),
              "# Chapter 5 Conclusion", hdr, para(350),
              "# References", "Smith, J. (2019). A paper.", "Jones, K. (2020). Another.",
              "# Appendix A", para(400)]
    return "\n".join(parts)


def test_split_chapters_types_and_exclusions():
    chs = split_chapters(thesis_md())
    assert [c["type"] for c in chs] == ["introduction", "literature_review", "methodology", "findings", "conclusion"]
    assert "Thanks to everyone" not in "".join(c["text"] for c in chs)          # front matter ignored
    assert not any("Smith, J. (2019)" in c["text"] for c in chs)                # references cut
    assert not any("RUNNING HEADER" in c["text"] for c in chs)                  # repeated header dropped only if >=5 repeats
    assert "Results and Discussion" in chs[3]["heading"]                        # classified as findings


def test_clean_removes_debris():
    txt = clean_thesis_text(thesis_md())
    for junk in ("| a | b |", "Table 3.1", "![img]", "\n42\n"):
        assert junk not in txt


def test_running_header_needs_five_repeats():
    md = "\n".join(["# Chapter 1 Introduction"] + [f"HDR\n{para(60, 'a' + str(i))}" for i in range(6)])
    assert "HDR" not in clean_thesis_text(md)
    md2 = "\n".join(["# Chapter 1 Introduction", "HDR", para(60), "HDR", para(60)])
    assert clean_thesis_text(md2).count("HDR") == 2


def test_units_drop_short_and_window_when_no_subheadings():
    chs = split_chapters(thesis_md())
    lit = split_units(chs[1])
    assert [u["heading"] for u in lit] == ["2.1 Background", "2.3 Debates"]      # 2.2 has only 50 words
    intro_units = split_units(chs[0])                                             # no sub-headings: windows
    assert intro_units and all(u["heading"].startswith("Chapter 1") for u in intro_units)
    assert all(n_words(u["text"]) >= 300 for u in intro_units)


def test_chunk_positions_bounds_and_paragraph_integrity():
    unit = {"heading": "H", "text": "\n\n".join(para(200, f"p{i}") for i in range(8))}
    chunks = chunk_unit(unit)
    assert chunks[0]["position"] == "open" and chunks[-1]["position"] == "close"
    assert all(c["position"] == "mid" for c in chunks[1:-1])
    assert all(c["word_count"] <= 650 for c in chunks)
    rebuilt = "\n\n".join(c["text"] for c in chunks)
    assert rebuilt == unit["text"]                                               # no paragraph split or lost
    single = chunk_unit({"heading": "H", "text": para(320)})
    assert len(single) == 1 and single[0]["position"] == "open"


def test_build_sections_shape():
    out = build_sections(thesis_md())
    assert {c["section_type"] for c in out["chunks"]} == {"introduction", "literature_review", "methodology", "findings", "conclusion"}
    assert all({"position", "chunk_idx", "text", "heading", "word_count"} <= set(c) for c in out["chunks"])
