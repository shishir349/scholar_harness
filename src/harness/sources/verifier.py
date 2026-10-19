"""Claim verification: does the cited source's abstract support the sentence that cites it? Report only."""
import json

from .. import llm
from ..text.citations import audit_citations, extract_citations, match_ref
from ..text.jsonparse import parse_json
from ..text.sentences import split_sentences
from ..text.strip import strip_for_style

VERDICTS = ("SUPPORTED", "PARTIAL", "UNSUPPORTED", "CANNOT_TELL")
BATCH = 12
MAX_CLAIMS = 40

SYSTEM = (
    "You check whether a cited source supports a sentence from a doctoral thesis. Judge ONLY from the abstract given; "
    "use no outside knowledge. Verdicts: SUPPORTED (the abstract states the claim or clearly implies it); "
    "PARTIAL (the abstract is related but the claim goes beyond it); UNSUPPORTED (the abstract contradicts the claim "
    "or is about a different topic); CANNOT_TELL (the abstract is silent on the specifics, such as numbers, "
    "mechanisms or quotations). Reply with ONLY a JSON array: "
    '[{"id": 1, "verdict": "SUPPORTED", "reason": "at most 20 words"}] with one entry per item.'
)


def find_cited_claims(text: str, refs: list[dict], cap: int = MAX_CLAIMS) -> list[dict]:
    """[{id, sentence, cite, ref}] for sentences whose citation matches a ref that has an abstract."""
    claims = []
    for sent in split_sentences(strip_for_style(text)):
        for cit in extract_citations(sent):
            ref = match_ref(cit, refs)
            if ref and ref.get("abstract"):
                claims.append({"sentence": sent, "cite": cit["raw"], "ref": ref})
    if len(claims) > cap:  # spread evenly instead of truncating the chapter's end
        step = len(claims) / cap
        claims = [claims[int(i * step)] for i in range(cap)]
    for i, c in enumerate(claims, 1):
        c["id"] = i
    return claims


def verify_claims(claims: list[dict], spend=None) -> list[dict]:
    """[{id, verdict, reason}] for every claim. Unparseable batch -> CANNOT_TELL for its claims."""
    results: dict[int, dict] = {}
    for i in range(0, len(claims), BATCH):
        batch = claims[i:i + BATCH]
        user = json.dumps([{"id": c["id"], "claim": c["sentence"], "abstract": c["ref"]["abstract"][:1500]} for c in batch])
        try:
            data = parse_json(llm.ask("haiku", SYSTEM, user, max_tokens=1200, spend=spend, temperature=0))   # same sentence, same verdict
        except Exception as e:  # noqa: BLE001
            from ..spend import BudgetExceeded
            if isinstance(e, BudgetExceeded):
                raise
            data = None
        got = {d.get("id"): d for d in data if isinstance(d, dict)} if isinstance(data, list) else {}
        for c in batch:
            d = got.get(c["id"], {})
            v = d.get("verdict") if d.get("verdict") in VERDICTS else "CANNOT_TELL"
            results[c["id"]] = {"id": c["id"], "verdict": v, "reason": str(d.get("reason") or "")[:200]}
    return [results[c["id"]] for c in claims]


def audit_chapter(content: str, refs: list[dict], spend=None, verify: bool = True) -> dict:
    """Citation audit (matched/unmatched/suspicious) plus optional claim verification."""
    text = strip_for_style(content)
    audit = audit_citations(extract_citations(text), refs)
    claim_checks: list[dict] = []
    if verify:
        claims = find_cited_claims(text, refs)
        verdicts = verify_claims(claims, spend) if claims else []
        by_id = {v["id"]: v for v in verdicts}
        claim_checks = [{"sentence": c["sentence"], "cite": c["cite"], "abstract": c["ref"]["abstract"][:600],
                         **by_id[c["id"]]} for c in claims]
    counts = {v: sum(1 for c in claim_checks if c["verdict"] == v) for v in VERDICTS}
    checked = len(claim_checks)
    return {
        "matched": audit["matched"], "unmatched": audit["unmatched"], "suspicious": audit["suspicious"],
        "claim_checks": claim_checks,
        "stats": {"cited": len(extract_citations(text)), "checked": checked, "supported": counts["SUPPORTED"],
                  "partial": counts["PARTIAL"], "unsupported": counts["UNSUPPORTED"], "cannot_tell": counts["CANNOT_TELL"],
                  "unsupported_rate": (counts["UNSUPPORTED"] / checked) if checked else 0.0,
                  "unmatched_count": len(audit["unmatched"])},
    }
