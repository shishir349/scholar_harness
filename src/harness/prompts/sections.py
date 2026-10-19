"""Section-generation prompts. Port of acdyon-scholar chapter.prompt.js (calls 1-3) and
sectionProfiles.js helpers. With every optional param absent the output is byte-identical to
Scholar's (tests/golden). Optional params are the harness additions."""
import json
import re
from functools import lru_cache
from pathlib import Path

from .base import ANTI_AI_RULES, BASE_PERSONA, NONE_DIRECTIVE

_FALLBACK_REFS = "Draw on seminal and recent peer-reviewed literature relevant to this domain and geography."


@lru_cache(maxsize=1)
def profiles() -> dict:
    return json.loads((Path(__file__).with_name("profiles.json")).read_text(encoding="utf-8"))


# ── profile detection (sectionProfiles.js:239-255) ──────────────────────────
# Order matters: first match wins, same as detectProfile.
_PATTERNS = [
    ("introduction", re.compile(r"\bintroduc")),
    ("literature_review", re.compile(r"lit(erature)?\s+review|theoretical\s+frame|conceptual\s+frame|theoretical\s+background|"
                                     r"related\s+work|prior\s+work|state\s+of\s+the\s+art")),
    ("methodology", re.compile(r"methodolog|research\s+design|data\s+collect|research\s+approach|"
                               r"materials?\s+and\s+methods?|experimental\s+(design|setup)")),
    ("findings", re.compile(r"\bfinding|result|\bempirical")),
    ("discussion", re.compile(r"\bdiscussion|analys")),
    ("conclusion", re.compile(r"conclusion|implication|contribution|recommendation|\breflection")),
    ("limitations", re.compile(r"limitation")),
    ("appendix", re.compile(r"appendix|appendic")),
]


def classify_title(title: str | None) -> str | None:
    """Profile id for a chapter title, or None when nothing matches."""
    t = (title or "").lower()
    for pid, rx in _PATTERNS:
        if rx.search(t):
            return pid
    return None


def detect_profile(chapter_title: str | None, chapter_number: int | None) -> dict:
    pid = classify_title(chapter_title)
    if pid:
        return profiles()[pid]
    if chapter_number == 1:
        return profiles()["introduction"]
    return profiles()["literature_review"]


def profile_block(p: dict) -> str:
    return f"WRITING REGISTER — {p['label'].upper()} (mode: {p['writingMode']}):\n{p['instruction']}"


def call2_note(p: dict) -> str:
    return f"Register ({p['writingMode']}): {p['call2Note']}"


def call3_note(p: dict) -> str:
    return f"Register ({p['writingMode']}): {p['call3Note']}"


def noise_block(p: dict) -> str:
    return p.get("noiseInstruction") or ""


def empirical_block(p: dict) -> str:
    return p.get("empiricalInstruction") or ""


# ── concept extraction (chapter.prompt.js:120) ──────────────────────────────
def extract_used_concepts(text: str | None) -> str:
    if not text:
        return "none yet"
    ws = [w for w in re.split(r'[\s,.\n;:()\[\]"]+', text) if len(w) > 2]
    freq: dict[str, int] = {}
    for w in ws[:-1]:  # JS loop stops before the last word
        if re.fullmatch(r"[A-Z][a-z]{2,}", w):
            freq[w] = freq.get(w, 0) + 1
    top = [w for w, c in sorted(freq.items(), key=lambda kv: -kv[1]) if c >= 2][:12]
    return ", ".join(top) if top else "See previously written text above"


# ── helpers ─────────────────────────────────────────────────────────────────
def _blk(x: str | None) -> str:
    return f"\n{x}\n" if x else ""


def _ref_line(r) -> str:
    if isinstance(r, dict):
        return r.get("apa_formatted") or r.get("apaFormatted") or r.get("raw_text") or r.get("apa") or ""
    return str(r)


def _refs_text(references) -> str:
    return "\n".join(_ref_line(r) for r in references) if references else _FALLBACK_REFS


def _feedback_block(feedback: str | None) -> str:
    return f"\n{feedback}\n" if feedback else ""


# ── call 1: theoretical foundation ──────────────────────────────────────────
def build_call1(*, thesis, chapter, section, words_target, references=None, memory_block=None,
                profile_block_=None, noise_block_=None, empirical_block_=None, ecology_block=None,
                temporal_block=None, exemplar_block=None, style_target_block=None, source_notes_block=None,
                citation_rules=None, feedback=None) -> str:
    ref_block = _refs_text(references)
    subs = section.get("subsections") or []
    sub_ctx = ("\nThis section encompasses the following sub-topics, which should be addressed within this section: "
               + " | ".join(f"{s['number']} {s['title']}" for s in subs)) if subs else ""
    extra_refs = "".join(f"\n\n{x}" for x in (source_notes_block, citation_rules) if x)
    blocks = "".join(_blk(b) for b in (memory_block, profile_block_, noise_block_, empirical_block_,
                                       ecology_block, temporal_block, exemplar_block, style_target_block))
    return f"""{BASE_PERSONA}

THESIS: "{thesis['title']}"
University: {thesis.get('university')} | Programme: {thesis.get('programme')}
Methodology: {thesis.get('methodology')} | Domain: {thesis.get('domain')}
Geographic Focus: {thesis.get('geographic_focus') or 'Not specified'}

CHAPTER {chapter['chapter_number']}: {chapter['title']}{sub_ctx}
{blocks}
YOUR TASK, OPENING PASSAGE:
Write the opening {words_target} words for section {section['number']}: {section['title']}

OBJECTIVE:
Establish the theoretical baseline or historical context for this section. Introduce the core analytical problem or conceptual tension. Build intellectual momentum. Do NOT resolve the argument: the passage must end on tension, contradiction, or unresolved complexity. No polished concluding sentence.

{ANTI_AI_RULES}

{NONE_DIRECTIVE}

REFERENCES:
{ref_block}{extra_refs}
{_feedback_block(feedback)}
---

Start with the APA Level 2 heading formatted as: {section['number']} {section['title']}
Write {words_target} words. No preamble. Begin now."""


# ── call 2: friction / critique ─────────────────────────────────────────────
def build_call2(*, thesis, chapter, section, chunk1, used_concepts, words_target, profile_note=None,
                exemplar_block=None, references_block=None, style_target_block=None, source_notes_block=None,
                citation_rules=None, feedback=None) -> str:
    note = f"{profile_note}\n" if profile_note else ""
    extra = "".join(f"\n{x}\n" for x in (exemplar_block, style_target_block, references_block, source_notes_block, citation_rules) if x)
    return f"""{BASE_PERSONA}

THESIS: "{thesis['title']}" | CHAPTER {chapter['chapter_number']}: {chapter['title']}
SECTION: {section['number']} {section['title']}
{note}
PREVIOUSLY WRITTEN TEXT, continue directly from where this ends:
---
{chunk1}
---

CONCEPTS ALREADY DISCUSSED, do not re-explain these:
{used_concepts}
{extra}
YOUR TASK, FRICTION AND CRITIQUE:
Generate the next {words_target} words. Continue EXACTLY from where the above text ends.

DO NOT summarise or repeat what was just written. DO NOT use transitional restating sentences.

INSTEAD, introduce intellectual friction appropriate to this section type:
- Challenge assumptions established in the opening
- Expose limitations, contradictions, or oversimplifications in the existing literature
- Discuss {thesis.get('geographic_focus') or 'emerging market'} complications and practical implementation constraints
- Reveal gaps between theoretical frameworks and operational reality
- Critique literature that presents overly neat or context-free conclusions
- Acknowledge unresolved debates instead of presenting stable consensus

{ANTI_AI_RULES}

{NONE_DIRECTIVE}
{_feedback_block(feedback)}
End on analytical tension: NOT on resolution.

Continue now ({words_target} words):"""


# ── call 3: synthesis / interpretation ──────────────────────────────────────
def build_call3(*, thesis, chapter, section, chunk1, chunk2, used_concepts, words_target, profile_note=None,
                exemplar_block=None, references_block=None, style_target_block=None, source_notes_block=None,
                citation_rules=None, feedback=None) -> str:
    note = f"{profile_note}\n" if profile_note else ""
    extra = "".join(f"\n{x}\n" for x in (exemplar_block, style_target_block, references_block, source_notes_block, citation_rules) if x)
    return f"""{BASE_PERSONA}

THESIS: "{thesis['title']}" | CHAPTER {chapter['chapter_number']}: {chapter['title']}
SECTION: {section['number']} {section['title']}
{note}
PREVIOUSLY WRITTEN TEXT, continue directly from where this ends:
---
{chunk1}

{chunk2}
---

CONCEPTS ALREADY DISCUSSED, do not repeat or re-explain:
{used_concepts}
{extra}
YOUR TASK, SYNTHESIS AND INTERPRETATION:
Generate the final {words_target} words for this section. Continue naturally from where the text above ends.

DO NOT repeat earlier concepts. DO NOT produce a generic summary. DO NOT close with "This section has demonstrated…" or any variation.

INSTEAD:
- Synthesise the theoretical foundation with the practical friction developed above
- Draw a defensible but non-clichéd academic interpretation calibrated to this section type
- Acknowledge what remains genuinely contested or unresolved
- Occasionally surface the researcher's analytical voice: "This study takes the position…", "What this analysis suggests…", "A closer reading reveals…"
- Position the argument to transition naturally toward the next section

STYLE: More reflective and synthesising than the previous passage. Stronger intellectual ownership. Allow one unusually short paragraph if it lands harder that way. Uneven sentence rhythm is preferable.

The ending should feel intellectually earned but slightly unresolved: thought-provoking, not neatly wrapped.

{ANTI_AI_RULES}

{NONE_DIRECTIVE}
{_feedback_block(feedback)}
Continue and conclude this section now ({words_target} words):"""
