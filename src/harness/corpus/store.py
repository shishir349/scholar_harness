"""SQLite + FTS5 corpus store (data/corpus.db)."""
import json
import sqlite3

from .. import config
from ..text.refs import looks_like_references

SCHEMA = """
CREATE TABLE IF NOT EXISTS theses(
  id INTEGER PRIMARY KEY, openalex_id TEXT UNIQUE NOT NULL, doi TEXT, title TEXT NOT NULL,
  author TEXT, university TEXT, year INTEGER, field_id TEXT NOT NULL, field_name TEXT NOT NULL, license TEXT,
  score REAL, pages INTEGER, holdout INTEGER NOT NULL DEFAULT 0, meta TEXT);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY, thesis_id INTEGER NOT NULL REFERENCES theses(id) ON DELETE CASCADE,
  field_name TEXT NOT NULL, section_type TEXT NOT NULL,
  position TEXT NOT NULL CHECK(position IN ('open','mid','close')),
  heading TEXT, chunk_idx INTEGER NOT NULL, word_count INTEGER NOT NULL, text TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_lookup ON chunks(section_type, position, field_name);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  heading, text, content='chunks', content_rowid='id', tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""


def connect(path=None) -> sqlite3.Connection:
    """`path` may be ':memory:'. Creates the schema when missing."""
    p = str(path if path is not None else config.db_path())
    if p != ":memory:":
        config.db_path().parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def upsert_thesis(con, t: dict) -> int:
    """t: openalex_id, title, field_id, field_name required. Returns the thesis id."""
    con.execute(
        """INSERT INTO theses(openalex_id, doi, title, author, university, year, field_id, field_name, license,
                              score, pages, holdout, meta)
           VALUES(:openalex_id,:doi,:title,:author,:university,:year,:field_id,:field_name,:license,
                  :score,:pages,:holdout,:meta)
           ON CONFLICT(openalex_id) DO UPDATE SET doi=excluded.doi, title=excluded.title, author=excluded.author,
             university=excluded.university, year=excluded.year, field_id=excluded.field_id,
             field_name=excluded.field_name, license=excluded.license, score=excluded.score,
             pages=excluded.pages, holdout=excluded.holdout, meta=excluded.meta""",
        {"doi": None, "author": None, "university": None, "year": None, "license": None, "score": None,
         "pages": None, "holdout": 0, "meta": None, **t,
         "holdout": int(bool(t.get("holdout"))),
         "meta": json.dumps(t["meta"]) if isinstance(t.get("meta"), (dict, list)) else t.get("meta")},
    )
    return con.execute("SELECT id FROM theses WHERE openalex_id=?", (t["openalex_id"],)).fetchone()["id"]


def delete_thesis_chunks(con, thesis_id: int) -> None:
    for r in con.execute("SELECT id, heading, text FROM chunks WHERE thesis_id=?", (thesis_id,)).fetchall():
        con.execute("INSERT INTO chunks_fts(chunks_fts, rowid, heading, text) VALUES('delete', ?, ?, ?)",
                    (r["id"], r["heading"], r["text"]))
    con.execute("DELETE FROM chunks WHERE thesis_id=?", (thesis_id,))


def insert_chunks(con, thesis_id: int, field_name: str, chunks: list[dict]) -> None:
    """chunks: {section_type, position, heading, chunk_idx, text} (word_count computed if absent).
    Bibliography text that leaked into a chapter (4% of the corpus, 19% of conclusions) is never stored."""
    for c in chunks:
        if looks_like_references(c["text"]):
            continue
        wc = c.get("word_count") or len(c["text"].split())
        cur = con.execute(
            "INSERT INTO chunks(thesis_id, field_name, section_type, position, heading, chunk_idx, word_count, text)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (thesis_id, field_name, c["section_type"], c["position"], c.get("heading"), c["chunk_idx"], wc, c["text"]))
        con.execute("INSERT INTO chunks_fts(rowid, heading, text) VALUES(?,?,?)",
                    (cur.lastrowid, c.get("heading") or "", c["text"]))


def drop_reference_chunks(con) -> int:
    """Remove bibliography chunks already in an older database (and from its search index). Returns how many."""
    n = 0
    for r in con.execute("SELECT id, heading, text FROM chunks").fetchall():
        if looks_like_references(r["text"]):
            con.execute("INSERT INTO chunks_fts(chunks_fts, rowid, heading, text) VALUES('delete', ?, ?, ?)",
                        (r["id"], r["heading"] or "", r["text"]))
            con.execute("DELETE FROM chunks WHERE id=?", (r["id"],))
            n += 1
    con.commit()
    return n


def replace_thesis_chunks(con, thesis_id: int, field_name: str, chunks: list[dict]) -> None:
    delete_thesis_chunks(con, thesis_id)
    insert_chunks(con, thesis_id, field_name, chunks)
    con.commit()


def set_meta(con, key: str, value) -> None:
    con.execute("INSERT INTO meta(key, value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)))
    con.commit()


def get_meta(con, key: str, default=None):
    r = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(r["value"]) if r else default


def random_holdout_chunk(con, section_type: str, position: str):
    """A random chunk from a holdout thesis (judge reference), or None."""
    r = con.execute(
        """SELECT c.text, c.heading, t.field_name, t.id AS thesis_id FROM chunks c JOIN theses t ON t.id=c.thesis_id
           WHERE t.holdout=1 AND c.section_type=? AND c.position=? ORDER BY random() LIMIT 1""",
        (section_type, position)).fetchone()
    return dict(r) if r else None
