"""Task 1.5: download, parse and gate candidate theses.
    python -m harness.corpus.acquire [--field slug] [--candidates-max N]
Resumable: a thesis folder with meta.json is skipped. Folder tree:
    data/raw/<field-slug>/<thesis-slug>/{thesis.pdf, thesis.md, sections.json, meta.json}  (failed theses are deleted)"""
import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

import httpx

from .. import config
from . import sections as sec
from .settings import (CANDIDATES_MAX, MIN_CHARS_PER_PAGE, MIN_TYPES_FOUND, MIN_WORDS_PER_TYPE, PAGES, PASS_TARGET,
                       REQUIRED_TYPES, USER_AGENT, slugify)

RAW = config.DATA / "raw"
MAX_BYTES = 80 * 1024 * 1024
DOWNLOAD_WORKERS = 6  # each in-flight download buffers up to MAX_BYTES in memory
# Most institutional repositories (DSpace, EPrints, bepress/Digital Commons) put the real PDF
# link in a "...citation_pdf_url" meta tag on the HTML landing page a bare DOI/handle resolves to.
_CITATION_PDF_RE = re.compile(
    rb'name=["\'][\w]*citation_pdf_url["\'][^>]*content=["\']([^"\']+)["\']'
    rb'|content=["\']([^"\']+)["\'][^>]*name=["\'][\w]*citation_pdf_url["\']', re.I)


def thesis_dir(field_slug: str, cand: dict):
    return RAW / field_slug / f"{slugify(cand['title'], 50)}-{cand['openalex_id'].lower()}"


def gate(md: str, pages: int, chars: int, toc: list | None = None) -> tuple[bool, str, dict | None]:
    """Hard quality gates -> (passed, reason, sections). sections = build_sections output when it passed."""
    if not PAGES[0] <= pages <= PAGES[1]:
        return False, f"pages {pages} outside {PAGES}", None
    if pages and chars / pages < MIN_CHARS_PER_PAGE:
        return False, f"{chars // max(pages, 1)} chars/page (scan?)", None
    built = sec.build_sections(md, toc)
    total_words = len(md.split())
    if total_words:
        # A real single chapter runs a few thousand words; one chapter entry (one heading, one
        # contiguous span) past this that also dominates the whole document means the heading
        # boundaries themselves broke and several real chapters got glued into one span. This is
        # about one chapter's own span, not the type total: several distinct real chapters that
        # happen to share a fallback type (e.g. several body chapters all counted as "findings")
        # are legitimate and must not trip this.
        biggest = max(built["chapters"], key=lambda c: c["words"], default=None)
        if biggest and biggest["words"] > 8000 and biggest["words"] > 0.45 * total_words:
            pct = round(100 * biggest["words"] / total_words)
            return False, f"chapter '{biggest['heading'][:40]}' looks mashed: {biggest['words']} words ({pct}% of document)", None
    words = {}
    for ch in built["chapters"]:
        words[ch["type"]] = words.get(ch["type"], 0) + ch["words"]
    good = [t for t in REQUIRED_TYPES if words.get(t, 0) >= MIN_WORDS_PER_TYPE]
    if len(good) < MIN_TYPES_FOUND:
        return False, f"only {len(good)} chapter types >= {MIN_WORDS_PER_TYPE} words: {good} (found types: {words})", None
    built["type_words"] = words
    return True, "ok", built


def download_pdf(urls: list[str], dest, client: httpx.Client) -> str | None:
    """First URL that yields a real PDF <= 80 MB; returns the URL or None.
    A landing page (HTML, not a PDF) is checked once for a citation_pdf_url meta tag
    and that URL is tried too - covers most DSpace/EPrints/bepress repository links."""
    queue = list(urls)
    seen = set()
    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            with client.stream("GET", url, follow_redirects=True, timeout=90) as r:
                if r.status_code != 200:
                    continue
                data = bytearray()
                for chunk in r.iter_bytes():
                    data += chunk
                    if len(data) > MAX_BYTES:
                        break
                if len(data) > MAX_BYTES:
                    continue
                if data.startswith(b"%PDF-"):
                    dest.write_bytes(bytes(data))
                    return url
                m = _CITATION_PDF_RE.search(bytes(data))
                if m:
                    pdf_url = (m.group(1) or m.group(2)).decode("utf-8", "replace").replace("&amp;", "&")
                    pdf_url = urljoin(str(r.url), pdf_url)  # repositories often emit a relative link
                    if pdf_url not in seen:
                        queue.append(pdf_url)
        except Exception:  # noqa: BLE001 - a malformed URL/redirect from an untrusted host must never kill the run
            continue
        finally:
            time.sleep(1)  # be polite between requests
    return None


def _parse_pdf_worker(pdf_path, md_path, toc_path, q) -> None:
    import fitz  # PyMuPDF (AGPL: corpus build only)
    import pymupdf4llm

    with fitz.open(pdf_path) as doc:
        pages = doc.page_count
        toc = doc.get_toc()
    if not PAGES[0] <= pages <= PAGES[1]:  # gate() rejects on page count first; skip the expensive parse
        md_path.write_text("", encoding="utf-8")
        toc_path.write_text("[]", encoding="utf-8")
        q.put((pages, 0))
        return
    try:
        texts =[p["text"] for p in pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)]
    except Exception:  # noqa: BLE001 - pymupdf4llm raises on some malformed page trees ("page not in document")
        texts = []  # plain per-page text: no markdown headings, but the TOC / "Chapter N" paths still work
        with fitz.open(pdf_path) as doc:
            for i in range(pages):
                try:
                    texts.append(doc[i].get_text())
                except Exception:  # noqa: BLE001
                    texts.append("")
    # Page markers let split_chapters_from_toc map the PDF's own table-of-contents page numbers
    # straight onto real text - far more reliable than guessing chapter boundaries from headings.
    md = "".join(f"\n\n<!-- page {i + 1} -->\n\n{t}" for i, t in enumerate(texts))
    md_path.write_text(md, encoding="utf-8")
    toc_path.write_text(json.dumps(toc), encoding="utf-8")
    q.put((pages, len(md)))


def parse_pdf(pdf_path, md_path, toc_path, timeout: int = 300) -> tuple[int, int]:
    """PDF -> markdown via pymupdf4llm, isolated in a subprocess: pymupdf4llm's layout
    model can OOM on a large PDF, and an OS kill must not take the whole acquire run down."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_parse_pdf_worker, args=(pdf_path, md_path, toc_path, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        raise RuntimeError("parse timed out")
    if not q.empty():
        return q.get()
    raise RuntimeError(f"parser process died (exit code {p.exitcode})")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--field")
    ap.add_argument("--candidates-max", type=int, default=CANDIDATES_MAX)
    args = ap.parse_args()
    fields = json.loads((config.DATA / "fields.selected.json").read_text(encoding="utf-8"))["fields"]
    RAW.mkdir(parents=True, exist_ok=True)
    rejected = RAW / "_rejected.jsonl"
    dl_fails: Counter = Counter()
    if rejected.exists():
        for line in rejected.read_text(encoding="utf-8").splitlines():
            r = json.loads(line) if line.strip() else {}
            if r.get("reason") == "download failed":
                dl_fails[r["openalex_id"]] += 1
    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        for f in fields:
            if args.field and f["slug"] != args.field:
                continue
            cands = json.loads((config.DATA / "candidates" / f"{f['slug']}.json").read_text(encoding="utf-8"))
            passed = sum(1 for c in cands if (thesis_dir(f["slug"], c) / "meta.json").exists())
            tried = 0
            # A candidate whose download already failed in 2+ earlier runs is hopeless (bot-walled or no PDF).
            todo = [c for c in cands if not (thesis_dir(f["slug"], c) / "meta.json").exists()
                    and dl_fails[c["openalex_id"]] < 2][:args.candidates_max]
            # Downloads dominate wall-clock (most fail slowly), so fetch them in parallel;
            # parse + gate stay sequential in candidate order.
            ex = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
            futs = {}
            for c in todo:
                d0 = thesis_dir(f["slug"], c)
                d0.mkdir(parents=True, exist_ok=True)
                futs[c["openalex_id"]] = ex.submit(download_pdf, c["pdf_urls"], d0 / "thesis.pdf", client)
            for c in todo:
                if passed >= PASS_TARGET:
                    break
                d = thesis_dir(f["slug"], c)
                tried += 1
                pdf, md_path, toc_path = d / "thesis.pdf", d / "thesis.md", d / "toc.json"
                used = futs[c["openalex_id"]].result()
                reason, built, pages = "download failed", None, None
                if used:
                    try:
                        pages, chars = parse_pdf(pdf, md_path, toc_path)
                        toc = json.loads(toc_path.read_text(encoding="utf-8"))
                        _, reason, built = gate(md_path.read_text(encoding="utf-8"), pages, chars, toc)
                    except Exception as e:  # noqa: BLE001 - a corrupt PDF must not stop the run
                        reason = f"parse error: {e}"
                    c = {**c, "pages": pages}
                if built:
                    (d / "sections.json").write_text(json.dumps(built), encoding="utf-8")
                    (d / "meta.json").write_text(json.dumps({**c, "field": f["name"], "field_id": f["id"], "field_slug": f["slug"],
                                                             "pdf_url": used}), encoding="utf-8")
                    passed += 1
                    print(f"  pass  {f['slug']:30} {c['title'][:60]}")
                else:
                    with rejected.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"field": f["slug"], "openalex_id": c["openalex_id"], "reason": reason}) + "\n")
                    for p in (pdf, md_path, toc_path):
                        p.unlink(missing_ok=True)
                    try:
                        d.rmdir()
                    except OSError:
                        pass
                    print(f"  skip  {f['slug']:30} {c['title'][:40]} -> {reason}")
            ex.shutdown(wait=True, cancel_futures=True)
            for c in todo[tried:]:  # prefetched but never needed: target reached first
                d = thesis_dir(f["slug"], c)
                for p in (d / "thesis.pdf", d / "thesis.md", d / "toc.json"):
                    p.unlink(missing_ok=True)
                try:
                    d.rmdir()
                except OSError:
                    pass
            print(f"{f['slug']:45} passers: {passed} (tried {tried})")
            if passed < 6:
                print(f"  SHORTFALL {f['name']}: {passed}/6. Rerun with --field {f['slug']} --candidates-max 80")


if __name__ == "__main__":
    main()
