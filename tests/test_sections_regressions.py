"""Regressions for evaluator bugs found on real rejected theses (each test mirrors one real failure)."""
from harness.corpus import sections as sec


def para(n: int, seed: str = "alpha") -> str:
    return " ".join(f"{seed}{i}" for i in range(n)) + "\n\n"


def types(md: str, toc=None) -> dict:
    out: dict = {}
    for ch in sec.build_sections(md, toc)["chapters"]:
        out[ch["type"]] = out.get(ch["type"], 0) + ch["words"]
    return out


def test_front_matter_appendix_heading_does_not_kill_the_thesis():
    md = ("# **LIST OF TABLES**\n\n# **TABLES IN APPENDIX**\n\n"
          f"# **1. INTRODUCTION**\n\n{para(900)}"
          f"# **2. LITERATURE REVIEW**\n\n{para(900)}"
          f"# **3. MATERIALS AND METHODS**\n\n{para(900)}"
          f"# **4. RESULTS AND DISCUSSIONS**\n\n{para(900)}"
          f"# **5. CONCLUSIONS**\n\n{para(900)}"
          "# **6. REFERENCES**\n\nsmith 2020\n\n# **7. APPENDICES**\n\n" + para(900))
    t = types(md)
    assert set(t) >= {"introduction", "literature_review", "methodology", "findings", "conclusion"}
    assert all(t[k] >= 800 for k in ("introduction", "literature_review", "methodology", "findings", "conclusion"))


def test_real_appendix_after_content_still_stops_the_thesis():
    md = f"# 1. Introduction\n\n{para(900)}# 5. Conclusions\n\n{para(900)}# Appendix A\n\n{para(5000, 'zz')}"
    assert sum(types(md).values()) < 2500


def test_bold_dotted_subsections_are_not_chapters():
    md = (f"# **1. INTRODUCTION**\n\n{para(900)}"
          f"# **3. MATERIALS AND METHODS**\n\n{para(500)}"
          f"# **3.3.2. Questionnaire and group discussion**\n\n{para(500, 'q')}"
          f"# **4. RESULTS**\n\n{para(900)}# **5. CONCLUSIONS**\n\n{para(900)}")
    t = types(md)
    assert t.get("methodology", 0) >= 1000  # sub-section text stays inside methodology
    assert "discussion" not in t


def test_numbered_list_items_and_captions_are_not_chapters():
    assert not sec._is_chapter_heading("1. The cover object is the carrier of the secret message and the stego object.")
    assert not sec._is_chapter_heading("Table 10. Number and Mean of each of goat classes")
    assert not sec._is_chapter_heading("2.1. Origin and Domestication of Goats")
    assert sec._is_chapter_heading("1. INTRODUCTION")
    assert sec._is_chapter_heading("Chapter 3")


def test_unnumbered_subheadings_stay_inside_their_chapter():
    md = (f"# **1. INTRODUCTION**\n\n{para(900)}"
          f"# **3. MATERIALS AND METHODS**\n\n{para(500)}"
          f"# **Ziquala district**\n\n{para(500, 'z')}# **Study site selection**\n\n{para(500, 's')}"
          f"# **4. RESULTS**\n\n{para(900)}# **5. CONCLUSIONS**\n\n{para(900)}")
    t = types(md)
    assert t["methodology"] >= 1500 and t["findings"] < 1000


def test_algorithm_steps_are_not_chapters():
    assert not sec._is_chapter_heading("2. Peak Signal-to-Noise Ratio (PSNR):")
    assert not sec._is_chapter_heading("6. If ( ≤ 256), then call IWSIM1 procedure")
    assert not sec._chapter_like("6. Encode each block as follows and then send it")
    assert sec._chapter_like("CONSTRUCTING THE ABORIGINE") and sec._chapter_like("4. Results and Analysis")


def test_bare_chapter_headings_take_their_title_from_the_next_line():
    md = (f"# **Chapter 1**\n\nIntroduction\n\n{para(900)}"
          f"# **Chapter 2**\n\nLiterature Review\n\n{para(900)}"
          f"# **Chapter 3**\n\nMethodology\n\n{para(900)}"
          f"# **Chapter 4**\n\nResults\n\n{para(900)}"
          f"# **Chapter 5**\n\nConclusion\n\n{para(900)}")
    assert set(types(md)) >= {"introduction", "literature_review", "methodology", "findings", "conclusion"}


def test_untitled_numbered_chapters_fall_back_to_intro_body_conclusion():
    md = "".join(f"# **Chapter {n}**\n\n{para(900, f'w{n}x')}" for n in range(1, 6))
    t = types(md)
    assert t["introduction"] >= 800 and t["conclusion"] >= 800 and t["findings"] >= 2400
