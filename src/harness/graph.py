"""write_section: the LangGraph pipeline for one thesis section (Scholar's three-call pattern plus
exemplars, a source ledger, validation, one bounded revision and a claim report)."""
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from . import config, llm
from .corpus import retrieve
from .prompts import sections as P
from .sources.tools import _FIGURE, CITATION_RULES, facts_for, merge_refs, source_notes_for
from .sources.verifier import audit_chapter
from .spend import BudgetExceeded
from .text import cleanup, numbers
from .text.citations import audit_citations, extract_citations, first_author
from .text.jsonparse import parse_json
from .text.sentences import split_sentences
from .text.strip import strip_for_style
from .validate.features import _STOCK_RX, style_features
from .validate.overlap import exemplar_ngrams, overlap_report, passage_overlaps
from .validate.profile import avoid_hits, load_profile, plan_ratios, profile_block, signpost_counts
from .corpus.profile import _citation_counts
from .validate.gapfeats import CONNECTIVES, feats, short_sentences
from .validate.style import feature_stats, score_section, style_target

POSITIONS = ("open", "mid", "close")
N_EXEMPLARS = int(config.get("HARNESS_EXEMPLARS", "3"))   # excerpts per draft call; 4 cost ~3K input tokens x 3 calls. Set 2 or 4 to A/B it.
# Voice floors as a fraction of the real-thesis mean for the section type (measured in data/style_baseline.json).
# Below the floor a cheap Haiku patch (first person) or a free paragraph merge (variety) runs; never a Sonnet rewrite.
VOICE_FLOOR = {"first_person_per_100": 0.5, "para_len_cv": 0.8}
# One per paragraph, sampled without repeats, so paragraphs differ in how they argue and not only in length.
PARAGRAPH_MOVES = [
    "open on one specific study, case or number from the source notes and build outward from it",
    "set two sources against each other and say what differs between them (design, sample, measure)",
    "qualify one finding: say where it holds and where it stops holding",
    "define or sharpen a term the argument depends on, using how the sources use it",
    "raise the one question the evidence leaves open and answer it as far as the sources allow, in a sentence that does not start with \"What\"",
    "explain why one method or technology choice follows from the finding before it",
    "give your own reading of the sources taken together as a plain claim, opening on its subject and not on \"What\"",
    "state one figure, date or sample size from the source notes and say what it means for the argument",
]
# Added by the plan, not drawn at random: every passage gets one example paragraph (real theses 1.97 examples per 1000 words,
# AI drafts 0.55) and the first-person paragraphs its section type calls for (real: intro 27%, lit review 7%, method 36%, discussion 29%).
EXAMPLE_MOVE = ("work one plain example from ordinary practice through to what it implies, marked as an example and uncited; if it uses a "
                "number, state that number in the example and keep any total to simple arithmetic from it")
FP_MOVE = "describe one choice you made (a source, a method, a scope) and the reason for it, in your own first-person voice"
# plain signposts the plan can ask for, with what each one needs so it is never empty decoration
SIGNPOST_HINTS = {
    "for example": "followed by a concrete case from the source notes (never an invented one)",
    "for instance": "followed by a concrete case from the source notes (never an invented one)",
    "in addition": "to add a further point the sources support",
    "however": "to mark a real contrast with what came just before",
    "according to": "and then name a listed source",
    "based on": "and then name the evidence in the source notes",
    "as a result": "for a consequence the sources show",
    "in contrast": "to set one finding against another",
    "in particular": "to single out the case that matters most",
}
CITE_CEILING = 1.25  # x the real citation rate for the section type; above it a Haiku patch merges/trims citations (was 1.5; drafts cite ~1.8x)
SOFT_STYLE = {"citation_per_100", "first_person_per_100", "para_len_cv"}   # fixed by patch or merge, never a Sonnet rewrite

# Measured on saved drafts: AI drafts cite ~1.6x more than real theses, never write in the first person
# (real: ~0.6 per 100 words), write evenly sized paragraphs and too many very short sentences. Instructions alone
# only nudge these, but they are free.
VOICE_RULES = (
    "VOICE RULES (real doctoral prose differs from typical AI prose; follow these):\n"
    "- Write as the researcher who did this work: first person where a human author would (\"I chose\", \"I used\", \"I treat X as\", "
    "\"I decided\"), a position taken, choices explained. Never announce a claim (\"I argue that\", \"I contend\", \"this study takes the "
    "position\"); state the claim itself.\n"
    "- Cite sparingly: about one citation per two or three sentences, several related sources in one parenthesis, at the end of the "
    "claim they support. Do not copy or lightly reword a source note, and do not write one \"Author (year) traces / notes / offers ...\" "
    "sentence per source: group sources by the point they support and say what they show together. Most paragraphs need no citation "
    "at all: your own analysis, comparison and example are uncited. When a source is named in the sentence, use a result verb "
    "(found, measured, tested, compared, used), never traces / notes / offers / highlights.\n"
    "- Be concrete: name the instruments, datasets, cases and figures in the source notes (never invent one). Use an exact number "
    "where the plan or the notes give one, and no number you were not given. Where it makes a point "
    "clearer, give a plain uncited example from ordinary practice in this field (\"for example, a sensor bolted to a bridge bearing "
    "sees ...\"), never as a finding, a figure or a named study you were not given. Compare sources on what really differs "
    "(design, sample, measure, setting, date). Prefer plain, exact words to impressive ones.\n"
    "- Rhythm: an average sentence near 29 words, but spread wide: some of 10-15 words beside some of 40-55 (sentence length varies "
    "more in real theses than in AI drafts), a sentence under eight words about one in twenty, never several long ones in a row. Vary "
    "paragraph length. Do not use em dashes.\n"
    "- Open a paragraph with its subject (\"The ...\", \"In ...\") or a case, never with \"What ...\" or a sweeping statement. End it on its "
    "last concrete point, not on a verdict, a one-line restatement or a saying. Do not force every paragraph into the same shape "
    "(claim, citation, limitation, gap); take a position where the evidence supports one and name a limitation only when the "
    "sources show one.\n"
    "- Say each point once: check each sentence against the passage so far and against POINTS ALREADY MADE.\n"
    "- No machine tells: contrasts that argue with no one (\"not just X, but Y\"), forced triples, stacked hedges, stock words "
    "(pivotal, intricate, delve, showcase, underscore, interplay), \"serves as\" for \"is\", \", highlighting the importance of\" riders, "
    "formulaic connectives (\"Furthermore,\", \"The picture becomes more complex still\", \"remains genuinely contested\")."
)
_FORMULAIC = re.compile("|".join([
    r"\bthe (?:problem|picture|question|difficulty|issue|challenge) (?:becomes|grows|gets) (?:even )?"
    r"(?:clearer|more complex|more nuanced|sharper)(?: still)?",
    r"\bit is (?:important|worth) (?:to note|noting|emphasi[sz]ing) that\b",
    r"(?:^|(?<=[.!?] ))(?:Furthermore|Moreover|Additionally|In addition|Notably),",
    r"\b(?:plays?|played) an? (?:crucial|pivotal|vital|key|critical) role\b",
    r"\bnot only\b[^.;]{0,100}\bbut also\b",
    r"\btapestry\b|\bdelves? into\b|\bstands? as a testament\b|\bin the realm of\b",
    r"\b(?:genuinely|deeply|hotly) (?:contested|unsettled|unresolved|impressive)\b",
    r"\bin certain respects\b|\bdoes not settle the question\b|\bnot a new problem\b|\bdoes not speak with one voice\b",
    r"\bsuggestive rather than conclusive\b",
    # humanizer (github.com/blader/humanizer) patterns 1-5, 12, 13, 15, 18, trimmed to what theses rarely do on purpose
    r"\b(?:it|this|that)(?:['\u2019]s| is| was) not (?:just|only|merely|simply)\b[^.;]{0,100}[;,] (?:it|this|that)(?:['\u2019]s| is| was)\b",
    r"\bdoes not mean\b[^.]{0,120}\.\s+(?:It|This|That) means\b",
    r"(?:^|(?<=[.!?] ))(?:That|This) (?:is|was) (?:the real|what really|exactly why)\b[^.]{0,40}\.",
    r"\bThat distinction matters\b|\bRead that again\b|\bLet that sink in\b|\bThe message was clear\b",
    r"\bthe real question is\b|\bat its core\b|\bwhat really matters\b|\bthe heart of the matter\b|\bthe deeper issue\b",
    r"\blet['\u2019]s (?:dive|explore|break)\b|\bhere['\u2019]s what you need to know\b|\bwithout further ado\b",
    r"(?:^|(?<=[.!?] ))(?:To be clear|Don['\u2019]t get me wrong|This is not to say|One might be tempted|A tempting approach)\b",
    r"\bdelv(?:e|es|ed|ing)\b|\bpivotal\b|\bintricac(?:y|ies)\b|\bintricate\b|\bmeticulous(?:ly)?\b|\bvibrant\b",
    r"\bshowcas(?:e|es|ed|ing)\b|\bunderscor(?:e|es|ed|ing)\b|\binterplay\b|\bbolster(?:s|ed|ing)?\b|\bgarner(?:s|ed)?\b|\bdeep dive\b",
    r"\bsetting the stage for\b|\bindelible mark\b|\bevolving landscape\b|\bDespite (?:these|such|its) challenges\b|\bthe future looks bright\b",
    r"\b(?:serves|stands|functions|operates) as an?\b|\bboasts\b",
    r",\s+(?:highlighting|underscoring|showcasing|fostering|cultivating|emphasi[sz]ing) (?:the|its|a|how)\b",
    # cues a Sonnet judge gave for a harness draft: tidy colon triads, signposted transitions, "opens a path"
    r"(?:^|(?<=[.!?] ))(?:The problem emerges|This study takes the position|The gap between)\b",   # template openers
    r":\s+[^,.;:()]{3,40}, [^,.;:()]{3,40}, and [^,.;:()]{3,40}[.;]",
    r"(?:^|(?<=[.!?] ))(?:Taken together|In summary|In conclusion|Overall),|\bis worth (?:noting|pinning down|highlighting)\b|\bworth noting\b",
    r"\bwhat (?:these|those|such) \w+(?: \w+)? share\b|\bthe appeal is obvious\b|\bopens? (?:up )?a path towards?\b",
]), re.I | re.M)
# a sentence that opens with a narrative citation ("Smith (2010) found"); 3+ in a paragraph is a source-by-source chain
_CHAIN = re.compile(r"(?:^|(?<=[.!?] ))[A-Z][\w'\u2019-]+(?: et al\.)?(?: (?:and|&) [A-Z][\w'\u2019-]+)? \((?:19|20)\d\d[a-z]?\)", re.M)
# Scholar's own prompts seed the stock phrases the mined profile calls tells ("The issue becomes more complex when", "This study
# takes the position", "What this analysis suggests"), ask for "at least two short sentences" against the profile's "fewer than one in
# ten", and make every passage end unresolved (so each ends on a verdict line). Replaced only on the harness path; Scholar's
# prompts stay byte-identical for the golden tests. A text that is not found is a no-op.
_SEEDS = (
    ("- Mix very short sentences (under 8 words) with long analytical ones. Include at least two short sentences per response",
     "- Let sentence length vary: most sentences run 18-35 words, with an occasional short one where it lands a point"),
    ("- Use natural transitions instead: \"What matters here is\u2026\", \"In practice\u2026\", \"The issue becomes more complex when\u2026\", "
     "\"The literature rarely addresses\u2026\", \"That assumption does not hold in\u2026\", \"The problem emerges when\u2026\"",
     "- Use plain transitions (\"For example,\", \"However,\", \"In addition,\", \"As a result,\", \"In contrast,\") or none when the "
     "next sentence follows on its own"),
    ("the passage must end on tension, contradiction, or unresolved complexity. No polished concluding sentence.",
     "the passage must end on a specific point the next passage picks up. No concluding verdict sentence."),
    ("End on analytical tension: NOT on resolution.", "End on a specific point, not on a summary or a verdict."),
    ("- Occasionally surface the researcher's analytical voice: \"This study takes the position\u2026\", \"What this analysis suggests\u2026\", "
     "\"A closer reading reveals\u2026\"",
     "- State your own reading plainly, at most once in this passage and never as an announcement (\"I read the evidence as\u2026\", \"in my view\")"),
    ("Allow one unusually short paragraph if it lands harder that way. Uneven sentence rhythm is preferable.",
     "Paragraph lengths follow the plan at the end. Uneven sentence rhythm is preferable."),
    ("The ending should feel intellectually earned but slightly unresolved: thought-provoking, not neatly wrapped.",
     "End on the last concrete point of the argument, in a plain sentence: no aphorism, no quotation, no verdict line."),
    # lines that license drift, or ask for critique the sources do not show (an abstract cannot back "the field is contested")
    ("- Allow occasional conceptual drift before returning to the main argument",
     "- Keep each paragraph on its one point; a tangent needs a reason the reader can see"),
    ("- Slight imperfections in flow are preferable to mechanical polish\n", ""),
    ("- Expose limitations, contradictions, or oversimplifications in the existing literature",
     "- Point out a limitation or contradiction only where a source note or your own reasoning shows it, and say which"),
    ("- Critique literature that presents overly neat or context-free conclusions",
     "- Name the specific study whose setting or design limits its conclusion, instead of criticising \"the literature\" in general"),
    ("- Acknowledge unresolved debates instead of presenting stable consensus",
     "- Acknowledge an unresolved debate only where the sources show one"),
    ("- Acknowledge what remains genuinely contested or unresolved", "- Say what the sources leave open, once, in a concrete sentence"),
    ("Use the literature against itself", "Set the sources against each other where they differ"),
    ("foreground competing schools and unresolved debates", "foreground the differences the source notes show"),
)


def _plain(prompt: str) -> str:
    for old, new in _SEEDS:
        prompt = prompt.replace(old, new)
    return prompt


# Share of real-thesis paragraphs (40+ words) that carry a citation, by section type (4,300 real paragraphs counted from the corpus).
CITE_PARA_REAL = {"introduction": 0.24, "literature_review": 0.47, "methodology": 0.28, "discussion": 0.19}


def cite_para_targets(chunks: dict, ptype: str) -> list:
    """[(pos, idx)] of single-citation paragraphs to patch, enough (at most 4) to bring the share of paragraphs that carry a citation
    back to the real share, when it is more than 12 points above it (harness drafts: 63% against 47% real in a literature
    review). A paragraph that carries a figure keeps its citation: the figure comes from that source."""
    real = CITE_PARA_REAL.get(ptype, 0.3)
    paras = [(pos, i, p) for pos in POSITIONS for i, p in enumerate(cleanup.paras(chunks[pos])) if not p.startswith("#") and len(p.split()) >= 40]
    cited = [(pos, i, p) for pos, i, p in paras if extract_citations(p)]
    if len(paras) < 5 or len(cited) / len(paras) <= real + 0.06:
        return []
    drop = min(5, math.ceil(len(cited) - (real + 0.03) * len(paras)))
    one = sorted((-len(p.split()), pos, i) for pos, i, p in cited if len(extract_citations(p)) == 1 and not _FIGURE.search(p))
    return [(pos, i) for _, pos, i in one[:drop]]


# Scholar's "natural academic realism" lines license restating a point ("conceptual restatement is authentic", "allow natural
# repetition"), against the harness rule "say each point once". Its empirical-realism lines tell the writer to say that studies
# conflict or failed to replicate, and (findings, discussion) to report effect sizes: with only abstracts as evidence that is
# an invented claim. Both are replaced on the harness path; Scholar's prompts stay as they are.
_REPEAT_LICENCE = re.compile(r"restat|circle back|backward glance|natural repetition|recursive|redundant|over-explain|elaborate than strictly", re.I)
_GROUNDED = ("EVIDENCE: describe disagreement between sources, mixed findings or failed replications only where a SOURCE NOTE says so, "
             "naming the source. Where the notes agree, say what they agree on. Do not invent a contradiction to sound critical.")
_NO_RESULTS = ("EVIDENCE: report only results the thesis context supplies. If none are supplied, say what the analysis tests and do not "
               "state effect sizes, correlations, quotations or participant responses. Discuss partial or unexpected support only "
               "where results were given, and hedge in proportion to the evidence.")
_EVIDENCE = {"introduction": _GROUNDED, "literature_review": _GROUNDED, "findings": _NO_RESULTS, "discussion": _NO_RESULTS}


def _trim_noise(block: str) -> str:
    head, _, body = block.partition("\n")
    if not body:
        return block
    return head + "\n" + " ".join(x for x in split_sentences(body) if not _REPEAT_LICENCE.search(x))


def _that_limit(ptype: str, profile: dict | None) -> float | None:
    """'that' per 1000 words above which cleanup.thin_that runs: 1.3 x the real median for the section type (AI drafts ~2.3x)."""
    tg = (profile or {}).get("targets") or {}
    t = (tg.get(ptype) or tg.get("all") or {}).get("that")
    return 1.3 * t["p50"] if t and t.get("p50") else None


def _mix(facts: list, ptype: str) -> list:
    """Interleave the fact kinds so every passage gets some of each: source figures lead in a literature review (that is where real
    authors quote them), the thesis's own numbers (chapter, objectives, questions, brief) lead everywhere else."""
    by: dict = {}
    for f in facts:
        by.setdefault(f.get("kind", "source"), []).append(f)
    cycle = ("source", "source", "structure", "brief") if ptype == "literature_review" else ("structure", "source", "brief", "source")
    out = []
    while any(by.values()):
        out += [by[k].pop(0) for k in cycle if by.get(k)]
    return out


def _numeric(state: dict) -> list:
    """Facts that hold a number the draft may state: source figures (only when sources are in use), the thesis's own numbers, the brief.
    Name-only source facts carry no number and are left out."""
    return [f for f in (state.get("facts") or []) if f.get("kind", "source") in ("source", "structure", "brief")
            and (state.get("rules") or f.get("kind", "source") != "source")]


def _round(x: float, rng) -> int:
    """x rounded up or down at random in proportion, so small quotas (0.24 of a paragraph) still show up on average."""
    return int(x) + (rng.random() < x - int(x))


def _is_chain(para: str) -> bool:
    """Source-by-source summary: three sentences opening with a source, or two opening "Author (year) traces / notes / offers ..."."""
    return len(_CHAIN.findall(para)) >= 3 or sum(1 for s in split_sentences(para) if cleanup.REPORT.search(s)) >= 2


def _made(text: str) -> str:
    """Opening sentence of each paragraph written so far, so a later call does not make the same point again."""
    firsts = [s[0] for p in cleanup.paras(text) if not p.startswith("#") for s in [split_sentences(p)] if s]
    return ("\nPOINTS ALREADY MADE (do not make any of these again, in any wording):\n"
            + "\n".join("- " + " ".join(s.split()[:24]) for s in firsts[:10])) if firsts else ""


def _glue(chunks: dict) -> str:
    """open+mid+close with a paragraph break between chunks that end a sentence and a space where one was cut
    mid-sentence (plain concatenation produced "understood.Part" and "remainslargely")."""
    out = chunks["open"]
    for pos in ("mid", "close"):
        nxt = chunks[pos]
        if not out.strip() or not nxt.strip() or out[-1].isspace() or nxt[0].isspace():
            out += nxt
        elif out.rstrip()[-1] in ".!?)\"'\u201d\u2019":
            out += "\n\n" + nxt
        else:
            out += " " + nxt
    return out


def tell_count(text: str) -> int:
    """Machine-writing tells in `text`: stock phrases plus the humanizer patterns above. A free, objective
    measure to compare drafts on (the judge is another model; this is not)."""
    return sum(1 for _ in _FORMULAIC.finditer(text)) + len(_STOCK_RX.findall(text))
_DASH = re.compile(r"[ \t]*—[ \t]*|[ \t]+–[ \t]+")


def _tidy(text: str) -> str:
    """Em dashes (and spaced en dashes) are a strong machine tell; real theses almost never use them."""
    return re.sub(r",[ \t]*([.,;:!?])", r"\1", _DASH.sub(", ", text))


HARD_REASONS = ("cites a work not in REFERENCES", "wording copied from a style exemplar", "generic/stock AI phrase",
                "copied from a source's abstract")


def prune_flagged(by_pos: dict, limit: int = 8) -> dict:
    """At most `limit` flagged paragraphs go to one patch call: paragraphs with a hard problem (unlisted citation, copied
    wording, stock phrase) first, then those with the most reasons, then the longest. More than that and the reply no longer
    fits, the JSON is cut off and the whole patch is thrown away."""
    items = [f for pos in POSITIONS for f in by_pos.get(pos, [])]
    if len(items) <= limit:
        return by_pos
    key = lambda f: (-sum(any(h in r for h in HARD_REASONS) for r in f["reasons"]), -len(f["reasons"]), -len(f["text"].split()))
    keep = {id(f) for f in sorted(items, key=key)[:limit]}
    return {pos: [f for f in by_pos[pos] if id(f) in keep] for pos in POSITIONS if any(id(f) in keep for f in by_pos.get(pos, []))}


@dataclass
class Deps:
    spend: Any = None                              # harness.spend.Spend
    con: Any = None                                # sqlite corpus connection (None = no exemplars)
    fields: list = field(default_factory=list)     # data/fields.selected.json "fields"
    emit: Callable[[dict], None] = lambda ev: None  # NDJSON status events
    baseline: dict | None = None                   # style baseline override (tests)
    lookup: Callable | None = None                 # (surname, year, hint) -> real reference | None; None = do not resolve
    profile: dict | None = None                    # mined style profile override; None = data/style_profile.json, {} = none


class S(TypedDict, total=False):
    thesis: dict
    chapter: dict
    section: dict
    references: list
    sources: list
    by_section: dict
    flags: dict
    profile: dict
    words_target: int
    max_tokens: int
    refs: list                # merged library + ledger sources
    ref_lines: list           # APA strings shown to the model
    added_refs: list          # real works the writer cited from its own knowledge, verified and adopted
    notes: str
    facts: list               # [{text, cite}] figure sentences from this section's sources
    rules: str
    exemplars: dict
    blocks: dict
    chunks: dict
    reports: dict
    revisions: int
    problems: dict
    n_sections: int


def words_target(chapter: dict, section: dict, n_sections: int) -> tuple[int, int]:
    """Same sizing as thesis.controller.js:761-765."""
    total_pages = chapter.get("estimated_pages") or 20
    pages = section.get("pages") or math.ceil(total_pages / max(1, n_sections))
    per_section = math.ceil(pages * 250)
    wt = max(200, min(700, math.ceil(per_section / 3)))
    return wt, math.ceil(wt * 1.6)


def _ref_lines(refs: list) -> list[str]:
    out = []
    for r in refs:
        out.append(P._ref_line(r) if isinstance(r, dict) else str(r))
    return [x for x in out if x]


def build_graph(deps: Deps):
    def gen(prompt: str, max_tokens: int, shared: str = "") -> str:
        msgs = [HumanMessage(content=prompt)]
        if shared:  # ponytail: one cached system block for the 3 drafts; under ~1K tokens the API just skips caching
            msgs.insert(0, SystemMessage(content=[{"type": "text", "text": shared, "cache_control": {"type": "ephemeral"}}]))
        return llm.text_of(llm.chat("sonnet", max_tokens=max_tokens, spend=deps.spend).invoke(msgs))

    # ── nodes ───────────────────────────────────────────────────────────────
    def context(state: S):
        deps.emit({"type": "status", "value": "drafting"})
        chapter, section = state["chapter"], state["section"]
        flags = state.get("flags") or {}
        n_secs = state.get("n_sections") or len((chapter.get("toc_sections") or {}).get("sections", [])) or 1
        wt, mt = words_target(chapter, section, n_secs)
        profile = P.detect_profile(chapter.get("title"), chapter.get("chapter_number"))
        use_sources = bool(flags.get("sources", True)) and bool(state.get("sources"))
        refs = merge_refs(state.get("references") or [], state["sources"]) if use_sources else list(state.get("references") or [])
        return {
            "profile": profile, "words_target": wt, "max_tokens": mt, "refs": refs, "ref_lines": _ref_lines(refs),
            "notes": source_notes_for(section, state["sources"], state.get("by_section")) if use_sources else "",
            "rules": CITATION_RULES if use_sources else "",
            "facts": (facts_for(section, state["sources"], state.get("by_section")) if use_sources else [])
                     + numbers.structure_facts(state["thesis"], chapter, section, profile["id"]) + numbers.brief_facts(state["thesis"]),
            "revisions": 0, "chunks": {}, "exemplars": {},
        }

    def retrieve_node(state: S):
        flags = state.get("flags") or {}
        profile = state["profile"]
        if not flags.get("pool", True) or deps.con is None or profile["id"] == "appendix":
            return {"exemplars": {}}
        t, s, c = state["thesis"], state["section"], state["chapter"]
        ex = {}
        for pos in POSITIONS:
            ex[pos] = retrieve.get_exemplars(con=deps.con, fields=deps.fields, title=t.get("title", ""), domain=t.get("domain", ""),
                                             chapter_title=c.get("title", ""), section_title=s.get("title", ""),
                                             section_type=profile["id"], position=pos, limit=N_EXEMPLARS)
        return {"exemplars": ex}

    def blocks_for(state: S) -> dict:
        p = state["profile"]
        return {pos: retrieve.build_exemplar_block(state.get("exemplars", {}).get(pos, []), p["id"], pos) or None
                for pos in POSITIONS}

    def _ref_block(state: S) -> str | None:
        if not (state.get("rules") and state["ref_lines"]):
            return None
        return ("REFERENCES (cite these; a well-known foundational work you are certain of may also be cited):\n"
                + "\n".join(state["ref_lines"]))

    def prof() -> dict | None:
        return deps.profile if deps.profile is not None else load_profile()

    def _shared(state: S) -> str:
        """Blocks identical in all 3 draft calls (style target, voice rules, references, source notes, citation
        rules): sent once as a cached system message so calls 2 and 3 read them at ~10% price. The reference list
        is what lifts it over the ~1K-token minimum the API needs before it caches anything."""
        p = state["profile"]
        target = None
        if (state.get("flags") or {}).get("pool", True) and p["id"] != "appendix":
            target = "\n\n".join(x for x in (style_target(p["id"], deps.baseline), profile_block(p["id"], prof(), deps.baseline),
                                              VOICE_RULES) if x)
        return "\n\n".join(x for x in (target, _ref_block(state), state.get("notes"), state.get("rules")) if x)

    def _concrete_rules(state: S, which: str, k: int, rng, moves: list) -> list[str]:
        """Per-paragraph add-ons from the mined profile, so soft targets become exact instructions: which sentence opens
        with which plain signpost, which paragraph holds the example, which paragraphs speak in the first person, which carry an
        exact number (and where in the paragraph), and which put their citations in parentheses. Replaces entries of `moves`."""
        extra = [""] * k
        pr, ptype, idx = prof() or {}, state["profile"]["id"], POSITIONS.index(which)
        sp = pr.get("signposts") or {}
        ops = [o for o in sp.get("list") or [] if o in SIGNPOST_HINTS]
        if sp.get("real_rate") and ops:
            n_sign = max(1, round(1.3 * sp["real_rate"] * state["words_target"] / 29))      # 29 = real mean sentence length; 1.3 = some are not followed
            for j, i in enumerate(rng.sample(range(k), min(n_sign, k))):
                op = ops[(idx + j) % len(ops)]
                shown = op.capitalize() + ("" if op in ("according to", "based on") else ",")
                extra[i] += f' Within it, begin one sentence with "{shown}" {SIGNPOST_HINTS[op]}.'
        if EXAMPLE_MOVE not in moves:
            moves[rng.randrange(k)] = EXAMPLE_MOVE
        ex_i = moves.index(EXAMPLE_MOVE)
        extra[ex_i] += ' Introduce the example with "For example," or "such as".'
        free = [i for i in range(k) if i != ex_i]
        for i in rng.sample(free, min(len(free), _round(numbers.FP_PARA_TARGET.get(ptype, 0.22) * k, rng))):
            moves[i] = FP_MOVE
            extra[i] += (f' Write one sentence in the first person ("I chose", "I used"); verbs to draw on: {numbers.FP_VERBS}. '
                         'Never "I argue", "I contend" or "I maintain".')
        src_at: list = []
        mine = _mix(_numeric(state), ptype)[idx::3]                                   # a different fact per passage
        for i, f in zip(rng.sample(free, min(len(free), len(mine), max(1, _round(numbers.NUM_PARA_TARGET.get(ptype, 0.35) * k, rng)))), mine):
            place = "Open the paragraph with a sentence that states" if rng.random() < 0.5 else "Within it, state"
            if f.get("kind", "source") == "source":
                extra[i] += f' {place} this fact, exactly but in your own words: "{f["text"]}" {f["cite"]}.'
                src_at.append(i)
            else:
                extra[i] += f' {place} this, keeping the number exact and in your own words: {f["text"]}.'
        share = (pr.get("citations") or {}).get("narrative_share_real")
        if state.get("rules"):                                                         # citations only matter when sources are in use
            rest = [i for i in range(k) if i not in src_at]
            real = CITE_PARA_REAL.get(ptype, 0.3)                                       # real theses cite in about a quarter to a half of paragraphs
            none_at = rng.sample(rest, min(len(rest), k - 1, round((1 - real) * k)))
            for i in none_at:
                extra[i] += " This paragraph carries no citation: it is your own analysis, comparison or example."
            pool = [i for i in range(k) if i not in none_at]
            if share is not None and pool:
                for i in rng.sample(pool, min(len(pool), round((1 - min(share, 0.45)) * len(pool)))):
                    extra[i] += ' Cite in parentheses at the end of the sentence, e.g. (Lee, 2015); no "Lee (2015) argues" form in this paragraph.'
        return extra

    def _para_plan(state: S, which: str) -> str:
        """Paragraph lengths for this passage, drawn from a spread-out set so the draft is not evenly sized
        (real theses: paragraph-length variation ~0.7, drafts ~0.4). Seeded, so a rewrite gets the same plan."""
        if not (state.get("flags") or {}).get("pool", True) or state["profile"]["id"] == "appendix":
            return ""
        k = max(2, min(6, round(state["words_target"] / 140)))
        rng = random.Random(f"{state['section'].get('title')}|{which}")
        real = plan_ratios(state["profile"]["id"], prof())    # measured: paragraph length relative to the mean, real theses
        mult = rng.sample(real if real and len(real) >= k else [0.35, 0.6, 0.9, 1.2, 1.6, 2.1], k)
        total = sum(mult)
        sizes = [int(max(30, round(state["words_target"] * m / total, -1))) for m in mult]
        moves = rng.sample(PARAGRAPH_MOVES, k)       # a different move per paragraph: no fixed claim-citation-gap template
        extra = _concrete_rules(state, which, k, rng, moves)
        lines = [f"{i}. about {n} words: {m}{extra[i - 1]}" for i, (n, m) in enumerate(zip(sizes, moves), 1)]
        return ("\n\nPARAGRAPH PLAN for this passage (in order; the heading line is not counted). Follow the lengths so "
                "paragraphs differ clearly, and make each move from what the sources actually say, without turning it "
                "into a template. Use a number only where a line below or the source notes give one:\n" + "\n".join(lines))

    def _prompt(state: S, which: str, chunks: dict, feedback: str | None = None) -> str:
        harness_on = (state.get("flags") or {}).get("pool", True) and state["profile"]["id"] != "appendix"
        text = _build(state, which, chunks, feedback, harness_on)
        return _plain(text) if harness_on else text

    def _build(state: S, which: str, chunks: dict, feedback: str | None, harness_on: bool) -> str:
        p, common = state["profile"], dict(thesis=state["thesis"], chapter=state["chapter"], section=state["section"],
                                           words_target=state["words_target"])
        made = (lambda t: _made(t)) if harness_on else (lambda t: "")
        blocks = blocks_for(state)
        notes = rules = target = None   # shared blocks travel in the cached system message (see _shared)
        ref_block = None   # in the shared system block; call 1 only points at it
        refs_for_call1 = ["(the full list is under REFERENCES in the background notes above)"] if _ref_block(state) else state["ref_lines"]
        if which == "open":
            return P.build_call1(**common, references=refs_for_call1, profile_block_=P.profile_block(p),
                                 noise_block_=_trim_noise(P.noise_block(p)) if harness_on else P.noise_block(p),
                                 empirical_block_=(_EVIDENCE.get(p["id"]) if harness_on else None) or P.empirical_block(p),
                                 exemplar_block=blocks["open"], style_target_block=target,
                                 source_notes_block=notes, citation_rules=rules, feedback=feedback)
        if which == "mid":
            return P.build_call2(**common, chunk1=chunks["open"],
                                 used_concepts=P.extract_used_concepts(chunks["open"]) + made(chunks["open"]),
                                 profile_note=P.call2_note(p), exemplar_block=blocks["mid"], references_block=ref_block,
                                 style_target_block=target, source_notes_block=notes, citation_rules=rules, feedback=feedback)
        return P.build_call3(**common, chunk1=chunks["open"], chunk2=chunks["mid"],
                             used_concepts=P.extract_used_concepts(chunks["open"] + " " + chunks["mid"]) + made(chunks["open"] + "\n\n" + chunks["mid"]),
                             profile_note=P.call3_note(p), exemplar_block=blocks["close"], references_block=ref_block,
                             style_target_block=target, source_notes_block=notes, citation_rules=rules, feedback=feedback)

    def _draft(which: str):
        def node(state: S):
            chunks = dict(state.get("chunks", {}))
            chunks[which] = gen(_prompt(state, which, chunks) + _para_plan(state, which), state["max_tokens"], _shared(state))
            return {"chunks": chunks}
        return node

    def humanize(state: S):
        """One dedicated pass whose only job is prose style - not content, not citations, not length.
        Splitting this out of call1/2/3 (which already juggle content + citations + word count) gives
        style its own undivided attention instead of one instruction competing with several others."""
        flags = state.get("flags") or {}
        p = state["profile"]
        if not flags.get("humanize") or p["id"] == "appendix":
            return {}
        ex_texts = [e["text"] for pos in POSITIONS for e in state.get("exemplars", {}).get(pos, [])][:4]
        target = style_target(p["id"], deps.baseline)
        if not ex_texts and not target:
            return {}
        chunks = state["chunks"]
        # Already reads fine on the same measure the style gate uses: skip the rewrite rather than risk
        # a pass that isn't broken (a text-stats check, no extra API call).
        pre = score_section(_joined(chunks), p["id"], deps.baseline)
        if pre.get("available") and pre.get("passed"):
            return {}
        parts = ["Rewrite the three passages below (from a doctoral thesis draft) so they read like real doctoral "
                "prose - match the voice, rhythm and feel of the STYLE EXAMPLES below, the way a person who writes "
                "like that would phrase these same ideas. Keep the SAME content, meaning, citations and roughly the "
                "same length for each passage. Keep the heading line in the OPEN passage exactly as it is."]
        for i, ex in enumerate(ex_texts, 1):
            parts.append(f"[STYLE EXAMPLE {i}]\n\"\"\"\n{ex[:500]}\n\"\"\"")
        if target:
            parts.append(target + " (background pattern only - matching the STYLE EXAMPLES' feel matters more "
                                   "than hitting these numbers exactly.)")
        parts.append(VOICE_RULES)
        for pos in POSITIONS:
            parts.append(f"[{pos.upper()}]\n\"\"\"\n{chunks[pos]}\n\"\"\"")
        parts.append('Reply with ONLY this JSON: {"open": "...", "mid": "...", "close": "..."} - the three '
                     "rewritten passages, same order, nothing else.")
        data = parse_json(llm.ask("sonnet", "You edit thesis drafts precisely, matching a target prose style.",
                                  "\n\n".join(parts), max_tokens=state["max_tokens"] * 3, spend=deps.spend))
        if not isinstance(data, dict) or not all(isinstance(data.get(pos), str) and data[pos].strip() for pos in POSITIONS):
            return {}  # unparseable reply: keep the original chunks, never fail the run over a style pass
        return {"chunks": {pos: data[pos].strip() for pos in POSITIONS}}

    def _joined(chunks: dict) -> str:
        return _glue(chunks)

    def _unmatched(text: str, refs: list) -> list[str]:
        if not refs:
            return []
        dict_refs = [r for r in refs if isinstance(r, dict)]
        if not dict_refs:
            return []
        return [c["raw"] for c in audit_citations(extract_citations(text), dict_refs)["unmatched"]]

    tried: set = set()

    def _resolve(state: S, text: str) -> list[dict]:
        """Real works behind citations the writer gave from its own knowledge: each unmatched (Author, Year) is
        looked up by first author + year and adopted as a reference when it exists. Anything not found stays
        unmatched and goes through the normal fix-up."""
        if deps.lookup is None or not (state.get("flags") or {}).get("sources", True):
            return []
        dict_refs = [r for r in state["refs"] if isinstance(r, dict)]
        have = {r.get("openalex_id") for r in dict_refs} - {None}
        found = []
        for cit in audit_citations(extract_citations(text), dict_refs)["unmatched"]:
            surname = first_author(cit["author"])
            key = (surname.lower(), cit["year"])
            if key in tried or len(tried) >= 15:          # bounded look-ups per section
                continue
            tried.add(key)
            i = text.find(cit["raw"])
            hint = f"{state['thesis'].get('title') or ''} {text[max(0, i - 200):i]}"
            try:
                ref = deps.lookup(surname, cit["year"], hint)
            except Exception:  # noqa: BLE001 - a failed look-up just leaves the citation unverified
                continue
            if ref and ref.get("openalex_id") not in have:
                found.append({**ref, "resolved": True})
        return found

    def _floor(ptype: str, name: str) -> float | None:
        st = feature_stats(ptype, name, deps.baseline)
        return st["mean"] * VOICE_FLOOR[name] if st else None

    def _cv(chunks: dict) -> float | None:
        return (style_features(strip_for_style(_joined(chunks))) or {}).get("para_len_cv")

    def _vary(chunks: dict, floor: float) -> dict:
        """Free fix for evenly sized paragraphs: merge the shortest adjacent pair (max 2 merges, headings stay
        alone), keeping a merge only if it raises the variation."""
        chunks = dict(chunks)
        for _ in range(2):
            cv = _cv(chunks)
            if cv is None or cv >= floor:
                break
            best = None
            for pos in POSITIONS:
                paras = [x for x in re.split(r"\n\s*\n", chunks[pos]) if x.strip()]
                for i in range(len(paras) - 1):
                    if paras[i].lstrip().startswith("#") or paras[i + 1].lstrip().startswith("#"):
                        continue
                    n = len(paras[i].split()) + len(paras[i + 1].split())
                    if best is None or n < best[0]:
                        best = (n, pos, i, paras)
            if best is None:
                break
            _, pos, i, paras = best
            trial = {**chunks, pos: "\n\n".join(paras[:i] + [paras[i] + " " + paras[i + 1]] + paras[i + 2:])}
            if (_cv(trial) or 0) <= cv:
                break
            chunks = trial
        return chunks

    def _voice_targets(chunks: dict, ptype: str, features: dict | None) -> list:
        """[(pos, idx)] of paragraphs for a Haiku patch to give a first-person stance: as many as it takes to bring the share of
        paragraphs with a first-person word up to the real share for this section type (intro 27%, lit review 7%, method 36%,
        discussion 29%), or 2 when only the per-word rate is under half the real rate. Uncited, longer paragraphs first."""
        n = numbers.fp_gap(chunks, ptype)
        if not n:
            floor = _floor(ptype, "first_person_per_100")
            if not floor or not features or features.get("first_person_per_100", 0) >= floor:
                return []
            n = 2
        cands = []
        for pos in POSITIONS:
            for i, para in enumerate(cleanup.paras(chunks[pos])):
                w = len(para.split())
                if w >= 50 and not para.lstrip().startswith("#") and not numbers.has_first_person(para):
                    cands.append((len(extract_citations(para)), -w, pos, i))
        return [(pos, i) for _, _, pos, i in sorted(cands)[:n]]

    def _cite_targets(chunks: dict, ptype: str, features: dict | None) -> list:
        """[(pos, idx)] of the 2 most citation-heavy paragraphs when the draft cites over CITE_CEILING x the real rate."""
        st = feature_stats(ptype, "citation_per_100", deps.baseline)
        if not st or not features or features.get("citation_per_100", 0) <= st["mean"] * CITE_CEILING:
            return []
        cands = []
        for pos in POSITIONS:
            for i, para in enumerate(x for x in re.split(r"\n\s*\n", chunks[pos]) if x.strip()):
                n = len(extract_citations(para))
                if n >= 3:
                    cands.append((-n, pos, i))
        return [(pos, i) for _, pos, i in sorted(cands)[:2]]

    def _style_targets(chunks: dict, text: str) -> tuple[list, list]:
        """([(pos, idx)] to get a plain signpost, [(pos, idx)] to move narrative citations into parentheses). Empty unless the
        whole draft misses the real rate by a wide margin: signposts below half of it, narrative citations 20 points above it."""
        pr = prof() or {}
        real_sign = (pr.get("signposts") or {}).get("real_rate")
        real_nar = (pr.get("citations") or {}).get("narrative_share_real")
        paras = [(pos, i, para) for pos in POSITIONS for i, para in enumerate(x for x in re.split(r"\n\s*\n", chunks[pos]) if x.strip())]
        sign, nar = [], []
        n_sign, n_sent = signpost_counts(text)
        if real_sign and n_sent >= 12 and n_sign / n_sent < 0.8 * real_sign:
            cands = sorted((-n, pos, i) for pos, i, para in paras for s, n in [signpost_counts(para)] if n >= 3 and s == 0)
            sign = [(pos, i) for _, pos, i in cands[:3]]
        n_nar, n_par = _citation_counts(text)
        if real_nar is not None and n_nar + n_par >= 5 and n_nar / (n_nar + n_par) > real_nar + 0.2:
            cands = sorted((-a, pos, i) for pos, i, para in paras for a, _ in [_citation_counts(para)] if a >= 2)
            nar = [(pos, i) for _, pos, i in cands[:2]]
        return sign, nar

    def _plain_targets(chunks: dict, text: str, ptype: str) -> dict:
        """Paragraphs for the patches that close the measured gaps, empty unless the whole draft is far outside the real range
        for this section type (mined into the profile): "that" overuse, no plain connectives, runs of very short sentences,
        "the field"/"the literature" as a character."""
        out: dict = {"that": [], "connect": [], "short": [], "fieldlit": []}
        tg = (prof() or {}).get("targets") or {}
        t = tg.get(ptype) or tg.get("all") or {}
        f = feats(text) if t else None
        if not f:
            return out
        paras = [(pos, i, para) for pos in POSITIONS for i, para in enumerate(x for x in re.split(r"\n\s*\n", chunks[pos]) if x.strip())]
        top = lambda items: [(pos, i) for _, pos, i in sorted(items)[:2]]
        if "that" in t and f["that"] > 1.8 * t["that"]["p50"]:
            out["that"] = top([(-n, pos, i) for pos, i, para in paras for n in [len(re.findall(r"\bthat\b", para, re.I))] if n >= 3])
        if "connectives" in t and t["connectives"]["p50"] > 1 and f["connectives"] < 0.5 * t["connectives"]["p50"]:
            rx = re.compile(r"\b(?:" + "|".join(CONNECTIVES) + r")\b", re.I)
            out["connect"] = top([(-ns, pos, i) for pos, i, para in paras for sh, ns in [short_sentences(para)] if ns >= 3 and not rx.search(para)])
        if "short_sent_share" in t and f["short_sent_share"] > max(0.14, 2 * t["short_sent_share"]["p50"]):
            out["short"] = top([(-sh, pos, i) for pos, i, para in paras for sh, ns in [short_sentences(para)] if sh >= 3])
        if "the_field_lit" in t and f["the_field_lit"] > max(1.0, 3 * t["the_field_lit"]["p50"]):
            out["fieldlit"] = top([(-n, pos, i) for pos, i, para in paras
                                   for n in [len(re.findall(r"\bthe (?:literature|field)\b", para, re.I))] if n >= 1])
        return out

    def _gap_targets(chunks: dict, ptype: str, text: str, facts: list, taken: set = frozenset()) -> tuple[list, list]:
        """([(pos, idx, fact)] paragraphs that should state an exact figure, [(pos, idx)] paragraphs that should hold an example),
        empty unless the draft is under 70% of the target share. A figure is only ever one the request or the source notes gave."""
        if not text:
            return [], []
        nums, exs = [], []
        n = numbers.number_gap(chunks, ptype)
        if n:
            low = text.lower()
            left = [f for f in _mix(facts, ptype) if not numbers.fact_used(f, low)]
            body = [(pos, i, para) for pos, i, para in numbers.body_paras(chunks) if not numbers.has_figure(para) and (pos, i) not in taken]
            for f in left[:n]:
                cited = f.get("kind", "source") == "source"
                body.sort(key=lambda t: (bool(extract_citations(t[2])) != cited, -len(t[2].split())))     # a cited fact goes where a source is already cited
                if not body:
                    break
                pos, i, _ = body.pop(0)
                nums.append((pos, i, f["text"] + (f" {f['cite']}" if cited else "")))
        if numbers.example_gap(chunks):
            used = taken | {(pos, i) for pos, i, _ in nums}
            body = sorted(((bool(extract_citations(para)), -len(para.split()), pos, i) for pos, i, para in numbers.body_paras(chunks)
                           if not cleanup.EXAMPLE.search(para) and (pos, i) not in used))
            exs = [(pos, i) for _, _, pos, i in body[:numbers.example_gap(chunks)]]
        return nums, exs

    def validate(state: S):
        deps.emit({"type": "status", "value": "validating"})
        chunks = state["chunks"]
        ptype0 = state["profile"]["id"]
        pool_on = (state.get("flags") or {}).get("pool", True) and ptype0 != "appendix"
        vfloor = _floor(ptype0, "para_len_cv") if pool_on else None
        if pool_on:
            chunks = cleanup.heal(chunks)             # chunk seams: "remains.The", an orphan lowercase fragment, a cut-off last sentence
        if vfloor:
            chunks = _vary(chunks, vfloor)
        text = _joined(chunks)
        added = _resolve(state, text)
        refs = state["refs"] + added
        pasted, rep = [], {}
        if pool_on:        # free: delete sentences pasted from a source abstract or restating a neighbour; list the rest for a patch
            chunks, pasted = cleanup.drop_repeats(chunks, [r["abstract"] for r in refs if isinstance(r, dict) and r.get("abstract")])
            chunks, polished = cleanup.polish(chunks, _that_limit(ptype0, prof()))   # free: stance, report lines, "What" openers, closers, boosters, "that"
            text = _joined(chunks)
            rep = cleanup.repeat_flags(chunks)
        ex_texts = [e["text"] for pos in POSITIONS for e in state.get("exemplars", {}).get(pos, [])]
        ptype = state["profile"]["id"]
        style = score_section(text, ptype, deps.baseline) if ptype != "appendix" else {"available": False, "reason": "appendix"}
        ov = overlap_report(text, ex_texts)
        unmatched = _unmatched(text, refs) if state.get("rules") or refs else []
        stiff = sum(1 for para in re.split(r"\n\s*\n", text) if _FORMULAIC.search(para))
        voice = _voice_targets(chunks, ptype, style.get("features")) if pool_on else []
        cited_p = cite_para_targets(chunks, ptype) if pool_on else []
        figs, exs = _gap_targets(chunks, ptype, text, _numeric(state), set(cited_p)) if pool_on else ([], [])
        cites = _cite_targets(chunks, ptype, style.get("features")) if style.get("available") and pool_on else []
        chains = sum(1 for para in re.split(r"\n\s*\n", text) if _is_chain(para)) if pool_on else 0
        avoid = sum(1 for para in re.split(r"\n\s*\n", text) if len(avoid_hits(para, prof())) >= 2) if pool_on else 0
        signposts, narrative = _style_targets(chunks, text) if pool_on else ([], [])
        plain = _plain_targets(chunks, text, ptype) if pool_on else {}
        problems = {"voice": voice, "cites": cites, "chains": chains, "avoid": avoid, "signposts": signposts,
                    "narrative": narrative, "plain": plain, "unmatched": unmatched, "overlap": ov if ov["flagged"] else None,
                    "style": style if style.get("available") and not style.get("passed") else None,
                    "formulaic": stiff if stiff >= 2 else 0,   # one stock phrase is normal prose; a pattern is not
                    "citedpara": cited_p, "copied": pasted, "repeat": rep.get("repeat", []), "stance": rep.get("stance", []),
                    "closer": rep.get("closer", []), "whatopen": rep.get("whatopen", []),
                    "numbers": figs, "examples": exs}
        reports = {"style": {k: v for k, v in style.items() if k != "features"}, "overlap": ov,
                   "citations": {"unmatched": unmatched}}
        grown = {"refs": refs, "ref_lines": state["ref_lines"] + [r["apa"] for r in added],
                 "added_refs": state.get("added_refs", []) + added} if added else {}
        first = (state.get("reports") or {}).get("first_problems") or {
            k: (len(v) if isinstance(v, (list, tuple)) else bool(v)) if k != "style" else [f["feature"] for f in (v or {}).get("flags", [])]
            for k, v in problems.items()}
        return {"chunks": chunks, "reports": {**state.get("reports", {}), **reports, "first_problems": first},
                "problems": problems, **grown}

    def route(state: S):
        flags = state.get("flags") or {}
        p = state["problems"]
        bad = bool(p["unmatched"] or p["overlap"] or p["style"] or p.get("formulaic") or p.get("voice") or p.get("cites") or p.get("chains") or p.get("avoid")
                   or p.get("signposts") or p.get("narrative") or any((p.get("plain") or {}).values())
                   or p.get("citedpara") or p.get("copied") or p.get("repeat") or p.get("stance") or p.get("closer") or p.get("whatopen")
                   or p.get("numbers") or p.get("examples"))
        return "revise" if bad and flags.get("revise", True) and state["revisions"] == 0 else "verify"

    def _worst(state: S) -> str:
        """Passage most responsible for the problems: unmatched citations, then copied wording, then stock phrases."""
        ex_texts = [e["text"] for pos in POSITIONS for e in state.get("exemplars", {}).get(pos, [])]
        best, best_key = "mid", (0, 0, 0)
        for pos in POSITIONS:
            t = state["chunks"][pos]
            key = (len(_unmatched(t, state["refs"])), int(overlap_report(t, ex_texts)["flagged"]), len(_STOCK_RX.findall(t)))
            if key > best_key:
                best, best_key = pos, key
        return best

    def _flagged_paragraphs(state: S) -> dict[str, list[dict]]:
        """{pos: [{idx, text, reasons}]} for paragraphs carrying a citation/copied-wording/stock-phrase
        problem - localizable spots a patch can fix without touching the rest of the passage."""
        ex_texts = [e["text"] for pos in POSITIONS for e in state.get("exemplars", {}).get(pos, [])]
        ex_grams = exemplar_ngrams(ex_texts)
        out: dict[str, list[dict]] = {}
        for pos in POSITIONS:
            text = state["chunks"][pos]
            unmatched_here = set(_unmatched(text, state["refs"]))
            formulaic_on = bool(state["problems"].get("formulaic"))
            flagged = []
            for i, para in enumerate(p for p in re.split(r"\n\s*\n", text) if p.strip()):
                reasons = []
                if unmatched_here and any(c["raw"] in unmatched_here for c in extract_citations(para)):
                    reasons.append("cites a work not in REFERENCES")
                if passage_overlaps(para, ex_grams):
                    reasons.append("wording copied from a style exemplar")
                if _STOCK_RX.search(para):
                    reasons.append("contains a generic/stock AI phrase")
                m = _FORMULAIC.search(para) if formulaic_on else None
                if m:
                    reasons.append(f'contains the formulaic phrase "{m.group(0)}": replace just that phrase or sentence '
                                   "with a specific, direct point and leave the rest of the paragraph as it is")
                if (pos, i) in set(state["problems"].get("voice") or []):
                    reasons.append("has no voice of its own: rewrite so it includes one sentence in the researcher's "
                                   "first person about their own choice or reading (\"I chose\", \"I treat X as\", \"my reading\"), in plain words, "
                                   "as a plain statement and not an announcement, and leave the rest as it is")
                at = lambda key: {(a, b): c for a, b, c in state["problems"].get(key) or []}
                if (pos, i) in at("repeat"):
                    reasons.append(f'restates a point already made earlier ("{at("repeat")[(pos, i)]}..."): delete the restating sentence, or '
                                   "replace it with a new, specific point that moves the argument on; keep every citation that is still used")
                if (pos, i) in at("stance"):
                    reasons.append('announces its claim ("I argue", "I contend", "this study takes the position"): state the claim directly '
                                   "with no announcement, keeping its meaning and citations")
                if (pos, i) in at("closer"):
                    reasons.append(f'ends on a summarising verdict ("{at("closer")[(pos, i)]}..."): cut that sentence, or replace it with the '
                                   "paragraph's last concrete point (a case, a figure from the notes, a named study); no aphorism, no quotation")
                if (pos, i) in at("whatopen"):
                    reasons.append('opens with "What ...": open on its subject ("The ...", "In ...") or on the case instead, same content')
                if (pos, i) in set(state["problems"].get("citedpara") or []):
                    reasons.append("carries a citation that may only decorate its point: if the point is your own reasoning or common "
                                   "knowledge, state it without a citation; keep a citation only where the claim comes from that source")
                if (pos, i) in set(state["problems"].get("copied") or []):
                    reasons.append("contains a sentence copied from a source's abstract: restate that point in your own words, inside your own "
                                   "sentence, and keep its citation")
                if (pos, i) in at("numbers"):
                    reasons.append(f'states no exact figure: work this in with one sentence, in your own words, keeping its number exact '
                                   f'(and its citation if it has one): {at("numbers")[(pos, i)]}. Add no other number')
                if (pos, i) in set(state["problems"].get("examples") or []):
                    reasons.append('has no example: add one or two sentences with a plain example from ordinary practice, introduced with '
                                   '"for example" or "such as", uncited; no named study and no figure you were not given')
                pl = state["problems"].get("plain") or {}
                if (pos, i) in set(pl.get("that") or []):
                    reasons.append('uses "that" ' + str(len(re.findall(r"\bthat\b", para, re.I))) + ' times: drop "that" wherever the '
                                   "sentence reads fine without it, and keep everything else as it is")
                if (pos, i) in set(pl.get("connect") or []):
                    reasons.append("links its claims without plain connectives: join two of them with also / however / therefore / thus "
                                   "where the logic allows; do not add facts or change any citation")
                if (pos, i) in set(pl.get("short") or []):
                    reasons.append("has several sentences under ten words: join each to a neighbouring sentence with a comma, \"and\" or "
                                   "\"which\", keeping every fact and citation")
                if (pos, i) in set(pl.get("fieldlit") or []):
                    reasons.append('treats "the field" or "the literature" as a person: say who found or argued it by naming the studies '
                                   "or authors, and keep every citation")
                if (pos, i) in set(state["problems"].get("signposts") or []):
                    reasons.append("has no plain signpost: begin one sentence with a plain one where it fits (\"For example,\" before a "
                                   "concrete case, \"However,\" before a real contrast, \"In addition,\" before a further point); do not "
                                   "add facts or change any citation")
                if (pos, i) in set(state["problems"].get("narrative") or []):
                    reasons.append("cites almost only in the narrative form: move about half of the \"Smith (2010) found X\" "
                                   "citations into parentheses at the end of their sentence (\"X (Smith, 2010)\"); keep every "
                                   "citation and claim")
                hits = avoid_hits(para, prof()) if state["problems"].get("avoid") else []
                if len(hits) >= 2:
                    reasons.append("uses phrasing real theses almost never use (" + ", ".join(f'"{h}"' for h in hits) + "): say each "
                                   "point plainly and concretely instead, with a \"for example\" or \"such as\" where an example "
                                   "helps; keep the meaning and the citations")
                if state["problems"].get("chains") and _is_chain(para):
                    reasons.append("reads as a source-by-source chain: make the researcher the subject, group the sources "
                                   "by the point they support (\"several studies agree..., though X disputes it\"), and say what "
                                   "they show together; keep the same citations")
                if (pos, i) in set(state["problems"].get("cites") or []):
                    reasons.append("cites too many sources one after another: merge related sources into one "
                                   "parenthesis, drop the weakest, and turn at least one source-by-source sentence into the "
                                   "researcher's own synthesis; keep only works already cited in this passage")
                if reasons:
                    flagged.append({"idx": i, "text": para, "reasons": reasons})
            if flagged:
                out[pos] = flagged
        return out

    def _patch(state: S, by_pos: dict[str, list[dict]]) -> dict[str, str] | None:
        """One call rewrites every flagged paragraph in place (across all three chunks) and nothing
        else. None when the model's reply doesn't parse into exactly the right count of replacements -
        the caller then falls back to the full-chunk rewrite."""
        flat = [f for pos in POSITIONS for f in by_pos.get(pos, [])]
        need_refs = any("REFERENCES" in r for f in flat for r in f["reasons"])
        parts = ["Below are paragraphs from a doctoral thesis draft, each flagged with a specific problem. Rewrite "
                "ONLY the flagged text to fix its problem(s), in your own words, keeping the same length, meaning "
                "and position in the argument - it must still fit naturally where it sits. No commentary."]
        if need_refs and state["ref_lines"]:
            parts.append("REFERENCES (cite only these):\n" + "\n".join(state["ref_lines"]))
        for i, f in enumerate(flat, 1):
            parts.append(f"[{i}] Problem: {'; '.join(f['reasons'])}.\nPassage:\n\"\"\"\n{f['text']}\n\"\"\"")
        parts.append(f"Reply with ONLY a JSON array of {len(flat)} strings, one rewritten passage per item, same order.")
        budget = min(4000, max(state["max_tokens"], int(1.7 * sum(len(f["text"].split()) for f in flat)) + 300))   # the reply is all the paragraphs again
        data = parse_json(llm.ask("haiku", "You edit thesis drafts precisely and minimally.", "\n\n".join(parts),
                                  max_tokens=budget, spend=deps.spend))
        if not isinstance(data, list) or len(data) != len(flat):
            return None
        chunks, k = dict(state["chunks"]), 0
        for pos in POSITIONS:
            flagged = by_pos.get(pos)
            if not flagged:
                continue
            paras = [p for p in re.split(r"\n\s*\n", chunks[pos]) if p.strip()]
            for f in flagged:
                new = data[k]
                if isinstance(new, str) and new.strip():
                    paras[f["idx"]] = new.strip()
                k += 1
            chunks[pos] = "\n\n".join(paras)
        return chunks

    def revise(state: S):
        deps.emit({"type": "status", "value": "revising"})
        p = state["problems"]
        by_pos = prune_flagged(_flagged_paragraphs(state))
        # A pure rhythm/style problem (sentence-length variance, hedging density, ...) is a property of
        # the whole passage, not one paragraph, so it can't be localized: it forces the full rewrite path.
        st = p["style"]
        style_hard = bool(st) and not (st["flags"] and all(f["feature"] in SOFT_STYLE for f in st["flags"]))
        if by_pos and not style_hard:
            patched = _patch(state, by_pos)
            if patched is not None:
                return {"chunks": patched, "revisions": 1}
        hard = p["unmatched"] or p["overlap"] or style_hard
        if not hard:        # only the soft voice nudge was due and the patch failed: not worth a Sonnet rewrite
            return {"revisions": 1}
        lines = ["PREVIOUS ATTEMPT PROBLEMS (fix these in this rewrite):"]
        if p["unmatched"]:
            lines.append("- Cited works that are NOT in REFERENCES: " + ", ".join(p["unmatched"]) +
                         ". Remove them, or cite only works listed in REFERENCES.")
        if p["overlap"]:
            lines.append("- Wording was copied from the style exemplars. Write it entirely in your own words.")
        if p.get("formulaic"):
            lines.append("- Formulaic connectives and stock evaluations (\"The problem becomes clearer when\", "
                         "\"Furthermore,\", \"remains genuinely contested\"). Advance the argument with specific points.")
        if p["style"]:
            names = ", ".join(f["feature"].replace("_", " ") for f in p["style"]["flags"]) or "overall rhythm"
            lines.append(f"- Reads as machine-written ({names}). Vary sentence length, hedge where evidence is thin, "
                         "commit to a position, and avoid stock phrases and mechanical transitions.")
        feedback = "\n".join(lines)
        worst = _worst(state)
        chunks = dict(state["chunks"])
        # later passages continue from earlier ones, so a rewrite carries forward
        for pos in POSITIONS[POSITIONS.index(worst):]:
            chunks[pos] = gen(_prompt(state, pos, chunks, feedback if pos == worst else None) + _para_plan(state, pos),
                              state["max_tokens"], _shared(state))
        return {"chunks": chunks, "revisions": 1}

    def _locate(chunks: dict, sentence: str):
        """(pos, start, end) of `sentence` in the chunks: verbatim first, then tolerant of different whitespace."""
        for pos in POSITIONS:
            i = chunks[pos].find(sentence)
            if i >= 0:
                return pos, i, i + len(sentence)
        rx = re.compile(r"\s+".join(map(re.escape, sentence.split())))
        for pos in POSITIONS:
            m = rx.search(chunks[pos])
            if m:
                return pos, m.start(), m.end()
        return None

    def _drop_sentences(chunks: dict, bad: list[dict]) -> tuple[dict, int]:
        """Delete the sentences the checker still marks UNSUPPORTED: a claim no source backs does not stay in the text."""
        chunks, dropped = dict(chunks), 0
        for c in bad:
            loc = _locate(chunks, c["sentence"])
            if loc:
                pos, a, b = loc
                chunks[pos] = re.sub(r"(?m)^[ \t]+", "", re.sub(r"[ \t]{2,}", " ", chunks[pos][:a] + chunks[pos][b:]))
                dropped += 1
        return chunks, dropped

    def _fix_unsupported(state: S, checks: list[dict], verdicts: tuple = ("UNSUPPORTED",)) -> tuple[dict, int]:
        """One cheap call rewrites each sentence the checker marked UNSUPPORTED (or PARTIAL, when asked) so it stays
        within its source's abstract. A sentence that can't be found in the chunks is left alone."""
        bad = [c for c in checks if c["verdict"] in verdicts]
        if not bad:
            return state["chunks"], 0
        parts = ["Each sentence below cites a source but goes beyond what that source's abstract says. Rewrite ONLY "
                 "the sentence so it stays within the abstract (keep the same citation, similar length, same role in "
                 "the argument). If the abstract does not support the point at all, state it as the researcher's own "
                 "argument without a citation. No commentary."]
        for i, c in enumerate(bad, 1):
            parts.append(f"[{i}] Sentence: {c['sentence']}\nSource abstract: {c['abstract']}\nProblem: {c['reason']}")
        parts.append(f"Reply with ONLY a JSON array of {len(bad)} strings, one rewritten sentence per item, same order.")
        data = parse_json(llm.ask("haiku", "You edit thesis drafts precisely and minimally.", "\n\n".join(parts),
                                  max_tokens=state["max_tokens"], spend=deps.spend))
        if not isinstance(data, list) or len(data) != len(bad):
            return state["chunks"], 0
        chunks, fixed = dict(state["chunks"]), 0
        for c, new in zip(bad, data):
            loc = _locate(chunks, c["sentence"]) if isinstance(new, str) and new.strip() else None
            if loc:
                pos, a, b = loc
                chunks[pos] = chunks[pos][:a] + new.strip() + chunks[pos][b:]
                fixed += 1
        return chunks, fixed

    def verify(state: S):
        """Check every citing sentence against its source's abstract. Anything UNSUPPORTED is rewritten once (PARTIAL ones
        ride along), re-checked, and if it still fails it is deleted, then re-checked again, so the text that leaves here
        has no sentence the checker calls unsupported. Verdicts run at temperature 0 so a re-check agrees with this one."""
        deps.emit({"type": "status", "value": "verifying"})
        chunks = dict(state["chunks"])
        dict_refs = [r for r in state["refs"] if isinstance(r, dict)]
        claims: dict = {}
        fixed = dropped = 0
        try:
            audit = audit_chapter(_joined(chunks), dict_refs, deps.spend, verify=True)
            claims = {"stats": audit["stats"], "checks": audit["claim_checks"]}
            for action in (("rewrite", "delete", "delete", "delete") if (state.get("flags") or {}).get("revise", True) else ()):
                # the first pass also rewrites PARTIAL sentences (related but going past the abstract); deletion is for UNSUPPORTED only
                bad = [c for c in claims["checks"] if c["verdict"] in (("UNSUPPORTED", "PARTIAL") if action == "rewrite" else ("UNSUPPORTED",))]
                if not bad:
                    break
                n = 0
                if action == "rewrite":
                    chunks, n = _fix_unsupported({**state, "chunks": chunks}, claims["checks"], ("UNSUPPORTED", "PARTIAL"))
                    fixed += n
                if n == 0:                                   # no rewrite (or it failed): remove what cannot be backed
                    bad = [c for c in bad if c["verdict"] == "UNSUPPORTED"]
                    chunks, n = _drop_sentences(chunks, bad)
                    dropped += n
                    if n == 0:
                        break                                # cannot find the sentence: report it, do not loop
                audit = audit_chapter(_joined(chunks), dict_refs, deps.spend, verify=True)
                claims = {"stats": audit["stats"], "checks": audit["claim_checks"]}
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001 - the report is optional, the text is not
            claims = {**claims, "error": str(e)}
        if fixed:
            claims["fixed_unsupported"] = fixed
        if dropped:
            claims["dropped_unsupported"] = dropped
        out = {"chunks": chunks} if (fixed or dropped) else {}
        return {**out, "reports": {**state["reports"], "claims": claims}}

    g = StateGraph(S)
    g.add_node("context", context)
    g.add_node("retrieve", retrieve_node)
    g.add_node("draft_open", _draft("open"))
    g.add_node("draft_mid", _draft("mid"))
    g.add_node("draft_close", _draft("close"))
    g.add_node("humanize", humanize)
    g.add_node("validate", validate)
    g.add_node("revise", revise)
    g.add_node("verify", verify)
    g.add_edge(START, "context")
    g.add_edge("context", "retrieve")
    g.add_edge("retrieve", "draft_open")
    g.add_edge("draft_open", "draft_mid")
    g.add_edge("draft_mid", "draft_close")
    g.add_edge("draft_close", "humanize")
    g.add_edge("humanize", "validate")
    g.add_conditional_edges("validate", route, {"revise": "revise", "verify": "verify"})
    g.add_edge("revise", "validate")
    g.add_edge("verify", END)
    return g.compile()


def write_section(request: dict, deps: Deps) -> dict:
    """request: {thesis, chapter, section, references, ledger:{sources,by_section}, flags, n_sections?}.
    -> {text, reports, revisions}"""
    ledger = request.get("ledger") or {}
    state = build_graph(deps).invoke({
        "thesis": request["thesis"], "chapter": request["chapter"], "section": request["section"],
        "references": request.get("references") or [], "sources": ledger.get("sources") or [],
        "by_section": ledger.get("by_section") or {}, "flags": request.get("flags") or {},
        "n_sections": request.get("n_sections"),
    }, config={"recursion_limit": 40})
    c = state["chunks"]
    text = c["open"] + c["mid"] + c["close"]
    if (request.get("flags") or {}).get("pool", True) and state["profile"]["id"] != "appendix":
        c = cleanup.heal(c)                       # the claim fix can leave a seam or a restated sentence too
        c, _ = cleanup.drop_repeats(c, [r["abstract"] for r in state.get("refs", []) if isinstance(r, dict) and r.get("abstract")])
        c, _ = cleanup.polish(c, _that_limit(state["profile"]["id"], deps.profile if deps.profile is not None else load_profile()))
        text = _tidy(_glue(c))
    return {"text": text, "reports": state["reports"], "revisions": state["revisions"],
            "added_refs": state.get("added_refs", [])}
