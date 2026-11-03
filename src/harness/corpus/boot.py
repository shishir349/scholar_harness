"""Boot-time corpus: no disk needed. Download the pack from a private URL, check its hash, build the db in /tmp.

Env: HARNESS_CORPUS_URL (secret signed URL), HARNESS_CORPUS_SHA256, HARNESS_DB (e.g. /tmp/corpus.db).
Never fatal: on any problem the server still starts, just without exemplars (run.py treats a missing db as "no corpus").
"""
import hashlib
import logging
import tempfile
from pathlib import Path

import httpx

from . import pack

log = logging.getLogger("harness.boot")


def _download(url: str, to: Path) -> None:
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with open(to, "wb") as f:
            for part in r.iter_bytes():
                f.write(part)


def ensure_corpus(url: str, sha256: str, dest: Path | str, fetch=_download) -> bool:
    """True when `dest` exists afterwards. Never logs the URL (it is a secret)."""
    dest = Path(dest)
    if dest.exists():
        return True
    if not url:
        return False
    if len(sha256) != 64:
        log.error("HARNESS_CORPUS_URL is set but HARNESS_CORPUS_SHA256 is not: refusing an unchecked download")
        return False
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "pack.jsonl.gz"
        try:
            fetch(url, f)
            if hashlib.sha256(f.read_bytes()).hexdigest() != sha256.lower():
                log.error("corpus pack hash mismatch: not used")
                return False
            pack.unpack(f, dest)
        except Exception as e:  # noqa: BLE001 - httpx errors can carry the URL; log the type only
            log.error("corpus boot failed: %s", type(e).__name__)
            return False
    log.info("corpus ready at %s", dest)
    return True
