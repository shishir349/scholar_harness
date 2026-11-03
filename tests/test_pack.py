"""pack/unpack round trip + boot download. Offline, tiny corpus."""
import hashlib

import pytest

from harness.corpus import boot, pack, retrieve, store


def _make(path):
    c = store.connect(path)
    t = store.upsert_thesis(c, dict(openalex_id="W1", title="T1", field_id="1", field_name="Computer Science", score=90))
    h = store.upsert_thesis(c, dict(openalex_id="W2", title="T2", field_id="1", field_name="Computer Science", score=80, holdout=1))
    store.insert_chunks(c, t, "Computer Science", [
        dict(section_type="literature_review", position="open", heading="H", chunk_idx=0, text="neural networks learn representations"),
        dict(section_type="literature_review", position="mid", heading="H2", chunk_idx=1, text="software engineering in teams")])
    store.insert_chunks(c, h, "Computer Science", [
        dict(section_type="literature_review", position="open", heading="H", chunk_idx=0, text="holdout neural networks text")])
    store.set_meta(c, "fields", [{"name": "Computer Science", "keywords": ["neural networks"]}])
    c.commit()
    return c


def test_round_trip_keeps_rows_ids_meta_and_search(tmp_path):
    src = _make(tmp_path / "a.db")
    out = tmp_path / "p.jsonl.gz"
    pack.pack(tmp_path / "a.db", out)
    dst_path = tmp_path / "sub" / "b.db"
    pack.unpack(out, dst_path)
    dst = store.connect(dst_path)
    for q in ("SELECT id, openalex_id, holdout FROM theses ORDER BY id",
              "SELECT id, thesis_id, position, heading, text FROM chunks ORDER BY id"):
        assert [tuple(r) for r in dst.execute(q)] == [tuple(r) for r in src.execute(q)]
    assert store.get_meta(dst, "fields") == store.get_meta(src, "fields")
    hits = dst.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'neural'").fetchall()
    assert len(hits) == 2          # search index rebuilt
    retrieve.clear_cache()


def test_unpack_refuses_to_overwrite_nothing_half_written(tmp_path):
    _make(tmp_path / "a.db")
    out = tmp_path / "p.jsonl.gz"
    pack.pack(tmp_path / "a.db", out)
    bad = tmp_path / "bad.gz"
    bad.write_bytes(out.read_bytes()[:50])
    dst = tmp_path / "b.db"
    with pytest.raises(Exception):
        pack.unpack(bad, dst)
    assert not dst.exists()        # failed unpack leaves no half-built db


def _packed(tmp_path):
    _make(tmp_path / "a.db")
    out = tmp_path / "p.jsonl.gz"
    pack.pack(tmp_path / "a.db", out)
    return out.read_bytes()


def test_boot_downloads_verifies_and_builds(tmp_path):
    data = _packed(tmp_path)
    dest = tmp_path / "boot" / "corpus.db"
    got = boot.ensure_corpus("http://x/pack", hashlib.sha256(data).hexdigest(), dest, fetch=lambda url, to: to.write_bytes(data))
    assert got is True and dest.exists()


def test_boot_rejects_wrong_hash_and_builds_nothing(tmp_path):
    data = _packed(tmp_path)
    dest = tmp_path / "boot" / "corpus.db"
    got = boot.ensure_corpus("http://x/pack", "0" * 64, dest, fetch=lambda url, to: to.write_bytes(data))
    assert got is False and not dest.exists()


def test_boot_skips_when_db_exists_or_no_url(tmp_path):
    dest = tmp_path / "corpus.db"
    dest.write_bytes(b"x")
    assert boot.ensure_corpus("http://x", "a" * 64, dest, fetch=lambda *a: 1 / 0) is True   # already there: no download
    assert boot.ensure_corpus("", "", tmp_path / "none.db") is False
    assert boot.ensure_corpus("http://x", "", tmp_path / "none.db", fetch=lambda *a: 1 / 0) is False  # url without hash: refuse


def test_boot_download_failure_is_not_fatal(tmp_path):
    def boom(url, to):
        raise OSError("network down")
    assert boot.ensure_corpus("http://x", "a" * 64, tmp_path / "c.db", fetch=boom) is False
