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
    assert call(url, "/health", token=None) == (200, {"ok": True})


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
