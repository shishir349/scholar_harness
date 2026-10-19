"""Regressions for download_pdf bugs found on real repository landing pages."""
import httpx

from harness.corpus import acquire

PDF = b"%PDF-1.4\n" + b"x" * 200


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_relative_citation_pdf_url_is_resolved_against_the_landing_page(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire.time, "sleep", lambda s: None)

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/item/1":
            return httpx.Response(200, content=b'<meta name="citation_pdf_url" content="/file/File_abc">')
        if req.url.path == "/file/File_abc":
            return httpx.Response(200, content=PDF)
        return httpx.Response(404)

    dest = tmp_path / "t.pdf"
    with _client(handler) as c:
        used = acquire.download_pdf(["https://repo.example/item/1"], dest, c)
    assert used == "https://repo.example/file/File_abc" and dest.read_bytes() == PDF


def test_malformed_url_is_a_failed_download_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire.time, "sleep", lambda s: None)
    with _client(lambda req: httpx.Response(404)) as c:
        assert acquire.download_pdf(["/not/absolute", "ht!tp://bad"], tmp_path / "t.pdf", c) is None


def test_bepress_prefixed_meta_tag_and_html_entities(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire.time, "sleep", lambda s: None)

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/dissertations/9":
            return httpx.Response(200, content=b'<meta name="bepress_citation_pdf_url" content="https://repo.example/v.cgi?a=1&amp;b=2">')
        if req.url.path == "/v.cgi":
            return httpx.Response(200, content=PDF) if req.url.query == b"a=1&b=2" else httpx.Response(404)
        return httpx.Response(404)

    with _client(handler) as c:
        assert acquire.download_pdf(["https://repo.example/dissertations/9"], tmp_path / "t.pdf", c)
