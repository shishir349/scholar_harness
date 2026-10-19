import json
import os
import subprocess
import sys
from pathlib import Path

GOLD = json.loads((Path(__file__).parent / "golden" / "cases.json").read_text(encoding="utf-8"))["cases"][0]
ROOT = Path(__file__).resolve().parents[1]


def run(cmd, req, script=None, tmp=None, env_extra=None, raw=None):
    env = {**os.environ, "HARNESS_FAKE_LLM": "1", "OPENALEX_API_KEY": "", "HARNESS_DB": str(ROOT / "data" / "nope.db")}
    env.update(env_extra or {})
    if script is not None:
        p = tmp / "script.json"
        p.write_text(json.dumps(script), encoding="utf-8")
        env["HARNESS_FAKE_SCRIPT"] = str(p)
    proc = subprocess.run([sys.executable, "-m", "harness.run", *cmd], input=raw if raw is not None else json.dumps(req),
                          capture_output=True, text=True, env=env, cwd=ROOT, timeout=120)
    events = [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]   # every stdout line must be JSON
    return proc.returncode, events, proc.stderr


def base_req():
    return {"thesis": GOLD["thesis"], "chapter": {**GOLD["chapter"], "estimated_pages": 5},
            "section": {**GOLD["section"], "pages": 5.39}, "references": GOLD["references"],
            "flags": {"pool": False, "sources": False, "revise": True}}


def test_write_section_round_trip(tmp_path):
    code, ev, _ = run(["write-section"], base_req(), {"sonnet": ["one ", "two ", "three"]}, tmp_path)
    assert code == 0
    statuses = [e["value"] for e in ev if e["type"] == "status"]
    assert statuses[0] == "drafting" and "validating" in statuses and statuses[-1] == "verifying"
    final = ev[-1]
    assert final["type"] == "section" and final["text"] == "one two three" and final["revisions"] == 0
    assert set(final["reports"]) >= {"style", "overlap", "citations", "claims"} and "total_usd" in final["usage"]


def test_research_without_key_is_skipped_not_an_error(tmp_path):
    req = {"thesis": GOLD["thesis"], "chapter": GOLD["chapter"], "sections": [GOLD["section"]]}
    code, ev, _ = run(["research"], req, {"sonnet": []}, tmp_path)
    assert code == 0 and ev[0] == {"type": "status", "value": "researching"}
    assert ev[-1]["type"] == "ledger" and ev[-1]["sources"] == [] and ev[-1]["skipped"] == "no_key"


def test_audit(tmp_path):
    req = {"content": "<h2>1.1 Section under test</h2><p>Plain text (Ghost, 2015).</p>", "chapter": GOLD["chapter"],
           "thesis": GOLD["thesis"], "references": [], "claims": False}
    code, ev, _ = run(["audit"], req, {}, tmp_path)
    assert code == 0 and ev[-1]["type"] == "audit"
    assert ev[-1]["citations"]["unmatched"] == ["(Ghost, 2015)"] and "checks" in ev[-1]["structure"]


def test_malformed_json_exits_2_with_error_event(tmp_path):
    code, ev, _ = run(["write-section"], None, raw="{not json", tmp=tmp_path)
    assert code == 2 and ev == [{"type": "error", "message": ev[0]["message"]}]


def test_missing_key_and_unknown_command(tmp_path):
    code, ev, _ = run(["write-section"], {"thesis": {}}, {"sonnet": []}, tmp_path)
    assert code == 2 and "chapter" in ev[0]["message"]
    code, ev, _ = run(["bogus"], {})
    assert code == 2 and ev[0]["type"] == "error"


def test_model_failure_exits_1_with_error_event(tmp_path):
    code, ev, _ = run(["write-section"], base_req(), {"sonnet": []}, tmp_path)   # script exhausted -> model error
    assert code == 1 and ev[-1]["type"] == "error"
