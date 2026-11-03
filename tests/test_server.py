"""HTTP wrapper: auth, validation, happy path, errors never leak. Fake model, no network cost."""
import json
import threading
import urllib.error
import urllib.request

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from harness import llm, server

TOKEN = "t" * 24
TEXT = ("Lee et al. (2015) measured 120 nodes over 6 months, and the boards missed low frequency vibration across the whole span. "
        "That may not hold for short spans.")


@pytest.fixture()
def url():
    srv = server.make_server(TOKEN, 0, "127.0.0.1")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    llm.set_fake(None)


def call(url, path, body=None, token=TOKEN, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url + path, data=data, method="POST" if data is not None else "GET")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_needs_no_token(url):
    code, res = call(url, "/health", token=None)
    assert code == 200 and res["ok"] is True and isinstance(res["corpus"], bool)


def test_rejects_missing_or_wrong_token(url):
    assert call(url, "/refine", {"text": TEXT}, token=None)[0] == 401
    assert call(url, "/refine", {"text": TEXT}, token="x" * 24)[0] == 401


def test_refuses_to_start_without_token():
    with pytest.raises(SystemExit):
        server.make_server("", 0)


def test_bad_requests(url):
    assert call(url, "/refine", {"text": ""})[0] == 400
    assert call(url, "/refine", {"text": TEXT, "flags": {"tier": "gpt"}})[0] == 400
    assert call(url, "/refine", raw=b"not json")[0] == 400
    assert call(url, "/nope", {"text": TEXT})[0] == 404


def test_refines_and_keeps_protected_content(url):
    new = TEXT.replace("the boards missed", "the boards did not pick up")
    m = GenericFakeChatModel(messages=iter([AIMessage(content=new)]))
    llm.set_fake(lambda tier: m)
    code, res = call(url, "/refine", {"text": TEXT})
    assert code == 200 and res["changed"] and "did not pick up" in res["text"] and "Lee et al. (2015)" in res["text"]
    assert "usage" in res and res["chunks"][0]["status"] == "rewritten"


def test_model_failure_returns_original_not_500(url):
    class Boom(GenericFakeChatModel):
        def _generate(self, *a, **k):
            raise RuntimeError("anthropic 400 secret-detail")
    llm.set_fake(lambda tier: Boom(messages=iter([])))
    code, res = call(url, "/refine", {"text": TEXT})
    assert code == 200 and res["text"] == TEXT and not res["changed"]
    assert "secret-detail" not in json.dumps({k: v for k, v in res.items() if k != "chunks"})


# ---- /research, /write-section, /audit: same logic as the CLI, over HTTP ----
import tests.test_cli as cli  # noqa: E402  (reuse the golden request)


@pytest.fixture()
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENALEX_API_KEY", "")
    monkeypatch.setenv("HARNESS_DB", str(tmp_path / "nope.db"))


def test_new_endpoints_need_token(url, offline):
    for p in ("/research", "/write-section", "/audit"):
        assert call(url, p, {}, token=None)[0] == 401


def test_new_endpoints_reject_bad_requests(url, offline):
    for p in ("/research", "/write-section", "/audit"):
        code, res = call(url, p, {"thesis": {}})
        assert code == 400 and "required" in res["error"]


def test_research_without_key_is_skipped_not_an_error(url, offline):
    req = {"thesis": cli.GOLD["thesis"], "chapter": cli.GOLD["chapter"], "sections": [cli.GOLD["section"]]}
    code, res = call(url, "/research", req)
    assert code == 200 and res["type"] == "ledger" and res["sources"] == [] and res["skipped"] == "no_key"


def test_audit(url, offline):
    req = {"content": "<h2>1.1 Section under test</h2><p>Plain text (Ghost, 2015).</p>", "chapter": cli.GOLD["chapter"],
           "thesis": cli.GOLD["thesis"], "references": [], "claims": False}
    code, res = call(url, "/audit", req)
    assert code == 200 and res["type"] == "audit" and res["citations"]["unmatched"] == ["(Ghost, 2015)"]


def test_write_section(url, offline):
    class Scripted(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self
    m = Scripted(messages=iter([AIMessage(content=c) for c in ("one ", "two ", "three")]))
    llm.set_fake(lambda tier: m)
    code, res = call(url, "/write-section", cli.base_req())
    assert code == 200 and res["type"] == "section" and res["text"] == "one two three" and "total_usd" in res["usage"]


def test_write_section_model_failure_is_500_without_details(url, offline):
    class Boom(GenericFakeChatModel):
        def _generate(self, *a, **k):
            raise RuntimeError("anthropic 400 secret-detail")
        def bind_tools(self, tools, **kw):
            return self
    llm.set_fake(lambda tier: Boom(messages=iter([])))
    code, res = call(url, "/write-section", cli.base_req())
    assert code == 500 and "secret-detail" not in json.dumps(res)
