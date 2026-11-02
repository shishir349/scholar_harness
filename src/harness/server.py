"""Tiny HTTP wrapper around refine, for Render. Stdlib only.

    GET  /health                    -> {"ok": true}            (no auth, for Render's health check)
    POST /refine  {text, flags?}    -> same JSON as `harness.run refine` ('refined' event, minus 'type')
    Auth: header  Authorization: Bearer <HARNESS_TOKEN>.  The server refuses to start without HARNESS_TOKEN.
Stateless. No corpus needed. A section takes tens of seconds (one model call per ~350 words), so the caller sets a long timeout.
"""
import hmac
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config
from .refine import refine
from .spend import BudgetExceeded, Spend

log = logging.getLogger("harness.server")
MAX_BODY = 300_000  # bytes; a long chapter is ~60k
SLOTS = threading.BoundedSemaphore(int(config.get("HARNESS_MAX_CONCURRENT", "4")))  # ponytail: busy -> 429, caller keeps its own text


def run_refine(req: dict) -> dict:
    """-> response dict. Raises ValueError on a bad request, BudgetExceeded when the per-request cap is hit."""
    text = req.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("'text' is required and must be a non-empty string")
    flags = req.get("flags") or {}
    tier = flags.get("tier", "sonnet")
    if tier not in ("sonnet", "haiku"):
        raise ValueError("flags.tier must be 'sonnet' or 'haiku'")
    spend = Spend()
    res = refine(text, spend=spend, tier=tier, max_words=int(flags.get("max_words", 350)), polish=bool(flags.get("polish", True)))
    return {**res, "usage": spend.summary()}


class Handler(BaseHTTPRequestHandler):
    token = ""

    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _authed(self) -> bool:
        got = self.headers.get("Authorization", "")
        return hmac.compare_digest(got.encode(), f"Bearer {self.token}".encode())

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            return self._send(200, {"ok": True})
        self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/refine":
            return self._send(404, {"error": "not found"})
        if not self._authed():
            return self._send(401, {"error": "unauthorized"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = -1
        if n <= 0 or n > MAX_BODY:
            return self._send(413 if n > MAX_BODY else 400, {"error": f"body must be 1..{MAX_BODY} bytes"})
        try:
            req = json.loads(self.rfile.read(n))
            if not isinstance(req, dict):
                raise ValueError("body must be a JSON object")
        except (ValueError, UnicodeDecodeError) as e:
            return self._send(400, {"error": f"bad json: {e}"[:200]})
        if not SLOTS.acquire(blocking=False):
            return self._send(429, {"error": "busy"})
        try:
            self._send(200, run_refine(req))
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except BudgetExceeded:
            self._send(402, {"error": "per-request budget reached"})
        except Exception:  # noqa: BLE001 - never leak internals; the caller falls back to its own text
            log.exception("refine failed")
            self._send(500, {"error": "refine failed"})
        finally:
            SLOTS.release()

    def log_message(self, fmt, *args):  # request lines only, never bodies
        log.info("%s %s", self.address_string(), fmt % args)


def make_server(token: str, port: int, host: str = "0.0.0.0") -> ThreadingHTTPServer:
    if len(token) < 16:
        raise SystemExit("HARNESS_TOKEN must be set (16+ chars)")
    handler = type("H", (Handler,), {"token": token})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    srv = make_server(config.get("HARNESS_TOKEN"), int(os.environ.get("PORT", "8000")))
    log.info("harness refine server on :%s", srv.server_port)
    srv.serve_forever()


if __name__ == "__main__":
    main()
