"""CLI contract (the future Node adapter talks to this).

    python -m harness.run research|write-section|audit

Reads ONE JSON object on stdin. Writes NDJSON events on stdout (one JSON per line, flushed), logs on stderr.
Exit: 0 ok, 2 bad request, 1 failure. Stateless: no server, nothing kept between calls.
"""
import json
import logging
import os
import sys

from . import config, llm
from .corpus import openalex, store
from .graph import Deps, write_section
from .prompts.sections import detect_profile
from .sources import agent as source_agent
from .sources.verifier import audit_chapter
from .spend import BudgetExceeded, Spend
from .validate.structure import check_structure
from .validate.style import score_section, split_chapter_by_sections

log = logging.getLogger("harness.run")


class BadRequest(ValueError):
    pass


def _emit(ev: dict, out=None) -> None:
    out = out or sys.stdout
    out.write(json.dumps(ev, ensure_ascii=False) + "\n")
    out.flush()


def _need(req: dict, **types) -> None:
    for k, t in types.items():
        if k not in req or not isinstance(req[k], t):
            raise BadRequest(f"'{k}' is required and must be {t.__name__}")


def _install_fake() -> None:
    """Test-only: HARNESS_FAKE_LLM=1 + HARNESS_FAKE_SCRIPT=<json {sonnet:[...], haiku:[...]}> swaps in scripted models."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    script = json.loads(open(os.environ["HARNESS_FAKE_SCRIPT"], encoding="utf-8").read())

    class Scripted(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    def mk(items):
        return Scripted(messages=iter([AIMessage(**i) if isinstance(i, dict) else AIMessage(content=i) for i in items]))

    models = {"sonnet": mk(script.get("sonnet", [])), "haiku": mk(script.get("haiku", []))}
    llm.set_fake(lambda tier: models[tier])


def _load_fields(con) -> list[dict]:
    if con is not None:
        got = store.get_meta(con, "fields")
        if got:
            return got
    p = config.DATA / "fields.selected.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("fields", [])
    except (OSError, ValueError):
        return []


def _open_db():
    p = config.db_path()
    return store.connect(p) if p.exists() else None   # never create an empty corpus by accident


def cmd_research(req: dict) -> None:
    _need(req, thesis=dict, chapter=dict, sections=list)
    spend = Spend()
    _emit({"type": "status", "value": "researching"})
    out = source_agent.research_chapter(req["thesis"], req["chapter"], req["sections"], spend=spend)
    if out is None:
        out = {"sources": [], "by_section": {}, "skipped": "agent_error"}
    _emit({"type": "ledger", **out, "usage": spend.summary()})


def cmd_write_section(req: dict) -> None:
    _need(req, thesis=dict, chapter=dict, section=dict)
    flags = req.get("flags") or {}
    spend = Spend()
    con = _open_db() if flags.get("pool", False) else None
    deps = Deps(spend=spend, con=con, fields=_load_fields(con), emit=_emit,
                lookup=openalex.find_cited_work if openalex.enabled() and flags.get("sources") else None)
    res = write_section(req, deps)
    _emit({"type": "section", **res, "usage": spend.summary()})


def cmd_audit(req: dict) -> None:
    _need(req, content=str, chapter=dict, thesis=dict)
    spend = Spend()
    chapter, thesis = req["chapter"], req["thesis"]
    ptype = detect_profile(chapter.get("title"), chapter.get("chapter_number"))["id"]
    secs = chapter.get("toc_sections")
    secs = secs.get("sections", []) if isinstance(secs, dict) else (secs or [])
    style = [{"number": s["number"], "title": s["title"], **{k: v for k, v in score_section(s["text"], ptype).items() if k != "features"}}
             for s in split_chapter_by_sections(req["content"], secs)]
    structure = check_structure(req["content"], chapter, thesis, ptype)
    cit = audit_chapter(req["content"], [r for r in (req.get("references") or []) if isinstance(r, dict)], spend,
                        verify=bool(req.get("claims", True)))
    _emit({"type": "audit", "style": style, "structure": structure,
           "citations": {"matched": len(cit["matched"]), "unmatched": [c["raw"] for c in cit["unmatched"]],
                         "suspicious": cit["suspicious"]},
           "claims": {"stats": cit["stats"], "checks": cit["claim_checks"]}, "usage": spend.summary()})


COMMANDS = {"research": cmd_research, "write-section": cmd_write_section, "audit": cmd_audit}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(argv) != 1 or argv[0] not in COMMANDS:
        _emit({"type": "error", "message": f"usage: python -m harness.run {'|'.join(COMMANDS)}"})
        return 2
    try:
        req = json.loads(sys.stdin.read())
        if not isinstance(req, dict):
            raise BadRequest("request must be a JSON object")
        if os.environ.get("HARNESS_FAKE_LLM") == "1":
            _install_fake()
        COMMANDS[argv[0]](req)
        return 0
    except (BadRequest, json.JSONDecodeError) as e:
        _emit({"type": "error", "message": str(e)})
        return 2
    except BudgetExceeded as e:
        _emit({"type": "error", "message": str(e), "code": "budget_exceeded"})
        return 1
    except Exception as e:  # noqa: BLE001
        log.exception("harness failed")
        _emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
