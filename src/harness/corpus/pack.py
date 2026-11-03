"""Corpus <-> one gzipped JSONL file (no search index inside: it is rebuilt on unpack, which is why the pack is ~8x smaller).

    python -m harness.corpus.pack pack  [data/corpus.db] [corpus-pack.jsonl.gz]   -> prints sha256 + size
    python -m harness.corpus.pack unpack <pack> <db>

The pack holds third-party thesis text. Never commit it; ship it through a private store (see corpus/boot.py).
"""
import gzip
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

from . import store

TABLES = ("theses", "chunks", "meta")  # order matters: chunks point at theses


def pack(db: Path | str, out: Path | str) -> str:
    """-> sha256 hex of the written file."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        with gzip.open(out, "wt", encoding="utf-8", compresslevel=9) as f:
            for t in TABLES:
                for r in con.execute(f"SELECT * FROM {t} ORDER BY 1"):
                    f.write(json.dumps({"t": t, "r": dict(r)}, ensure_ascii=False) + "\n")
    finally:
        con.close()
    return hashlib.sha256(Path(out).read_bytes()).hexdigest()


def unpack(src: Path | str, db: Path | str) -> None:
    """Build `db` from a pack. Written to a temp file first: a bad pack never leaves a half-built db behind."""
    db = Path(db)
    db.parent.mkdir(parents=True, exist_ok=True)
    tmp = db.with_name(db.name + ".part")
    tmp.unlink(missing_ok=True)
    con = None
    try:
        con = store.connect(tmp)
        with gzip.open(src, "rt", encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                if o["t"] not in TABLES:
                    raise ValueError(f"unknown table {o['t']!r}")
                cols = list(o["r"])
                con.execute(f"INSERT INTO {o['t']}({','.join(cols)}) VALUES({','.join(':' + c for c in cols)})", o["r"])
        con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        con.commit()
        con.close()
        con = None
        os.replace(tmp, db)
    except BaseException:
        if con is not None:
            con.close()
        tmp.unlink(missing_ok=True)
        raise


def main(argv: list[str]) -> int:
    if argv[:1] == ["pack"]:
        db, out = (argv[1:] + ["data/corpus.db", "corpus-pack.jsonl.gz"][len(argv) - 1:])[:2]
        sha = pack(db, out)
        print(f"{out}  {Path(out).stat().st_size} bytes  sha256={sha}")
        return 0
    if argv[:1] == ["unpack"] and len(argv) == 3:
        unpack(argv[1], argv[2])
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
