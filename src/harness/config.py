"""Settings. .env is parsed by hand (no extra dependency); real environment wins."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env(ROOT / ".env")


def get(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


def sonnet_model() -> str:
    return get("SONNET_MODEL", "claude-sonnet-4-6")


def haiku_model() -> str:
    return get("HAIKU_MODEL", "claude-haiku-4-5-20251001")


def budget_usd() -> float:
    try:
        return float(get("HARNESS_BUDGET_USD", "2"))
    except ValueError:
        return 2.0


def db_path() -> Path:
    p = Path(get("HARNESS_DB", "data/corpus.db"))
    return p if p.is_absolute() else ROOT / p


DATA = ROOT / "data"
