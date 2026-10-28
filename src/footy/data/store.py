"""Local parquet store for normalised matches."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("FOOTY_ROOT", Path(__file__).resolve().parents[3]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MATCHES = DATA_DIR / "matches.parquet"
FIXTURES = DATA_DIR / "fixtures.parquet"
META = DATA_DIR / "meta.json"


def save_store(matches: pd.DataFrame, source: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(MATCHES, index=False)
    META.write_text(json.dumps({
        "source": source,
        "rows": int(len(matches)),
        "first_date": str(matches["date"].min().date()),
        "last_date": str(matches["date"].max().date()),
    }, indent=2), encoding="utf-8")


def load_store() -> pd.DataFrame:
    if not MATCHES.exists():
        raise FileNotFoundError(f"{MATCHES} not found - run `footy fetch` first")
    return pd.read_parquet(MATCHES)


def load_meta() -> dict:
    return json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
