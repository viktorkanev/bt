"""Downloaders for raw match + odds data.

Primary source: football-data.co.uk (free CSVs, results + match stats + odds
from Bet365, Pinnacle, market max/avg, *including closing odds* from 2019/20).

Fallback: a GitHub mirror of the same data (results, stats, Bet365 + market
max odds, 2000 -> present, 38 divisions), useful when football-data.co.uk is
not reachable from the environment.
"""

from __future__ import annotations

import io
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from .. import leagues
from .schema import normalise_football_data, normalise_mirror, finalise

log = logging.getLogger(__name__)

FD_BASE = "https://www.football-data.co.uk"
MIRROR_URL = (
    "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/"
    "main/data/Matches.csv"
)
TIMEOUT = 60
HEADERS = {"User-Agent": "footy-engine/0.1 (+https://github.com)"}


def _get(url: str) -> bytes:
    r = requests.get(url, timeout=TIMEOUT, headers=HEADERS)
    r.raise_for_status()
    return r.content


def _read_csv_bytes(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content), encoding="latin-1", on_bad_lines="skip",
                       low_memory=False)


def fd_season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season_start(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def fetch_football_data(
    raw_dir: Path,
    league_codes: list[str] | None = None,
    first_season: int = 2000,
    refresh_recent: int = 1,
) -> pd.DataFrame:
    """Download football-data.co.uk CSVs (cached on disk).

    Older seasons are downloaded once; the most recent ``refresh_recent``
    seasons are always re-downloaded so new results flow in.
    """
    raw_dir = Path(raw_dir) / "football-data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    league_codes = league_codes or list(leagues.LEAGUES)
    cur = current_season_start()
    # Fail fast when the host is unreachable (e.g. blocked by a proxy).
    _get(f"{FD_BASE}/mmz4281/{fd_season_code(cur - 1)}/E0.csv")
    frames: list[pd.DataFrame] = []
    for code in league_codes:
        lg = leagues.get(code)
        if lg.extra:
            path = raw_dir / f"{code}.csv"
            url = f"{FD_BASE}/new/{lg.fd_code or code}.csv"
            try:
                path.write_bytes(_get(url))
            except Exception as e:  # noqa: BLE001
                if not path.exists():
                    log.warning("skip %s: %s", url, e)
                    continue
            frames.append(normalise_football_data(_read_csv_bytes(path.read_bytes()), code))
            continue
        for y in range(first_season, cur + 1):
            path = raw_dir / f"{code}_{fd_season_code(y)}.csv"
            if not path.exists() or y > cur - refresh_recent:
                url = f"{FD_BASE}/mmz4281/{fd_season_code(y)}/{code}.csv"
                try:
                    path.write_bytes(_get(url))
                except Exception as e:  # noqa: BLE001
                    log.info("skip %s: %s", url, e)
                    continue
            try:
                frames.append(normalise_football_data(_read_csv_bytes(path.read_bytes()), code))
            except Exception as e:  # noqa: BLE001
                log.warning("could not parse %s: %s", path, e)
    if not frames:
        raise RuntimeError("no data downloaded from football-data.co.uk")
    return finalise(pd.concat(frames, ignore_index=True))


def fetch_mirror(raw_dir: Path, refresh: bool = True) -> pd.DataFrame:
    raw_dir = Path(raw_dir) / "mirror"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "Matches.csv"
    if refresh or not path.exists():
        path.write_bytes(_get(MIRROR_URL))
    raw = pd.read_csv(path, low_memory=False)
    return normalise_mirror(raw)


def fetch_fixtures(raw_dir: Path) -> pd.DataFrame:
    """Upcoming fixtures with current odds from football-data.co.uk."""
    raw_dir = Path(raw_dir) / "football-data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in ("fixtures.csv", "new_league_fixtures.csv"):
        try:
            content = _get(f"{FD_BASE}/{name}")
        except Exception as e:  # noqa: BLE001
            log.warning("fixtures download failed (%s): %s", name, e)
            continue
        (raw_dir / name).write_bytes(content)
        raw = _read_csv_bytes(content)
        raw.columns = [str(c).strip().lstrip("﻿") for c in raw.columns]
        if "Div" in raw.columns:
            for div, grp in raw.groupby("Div"):
                frames.append(normalise_football_data(grp, str(div)))
        elif "Country" in raw.columns:
            lookup = {lg.name.lower(): lg.code for lg in leagues.LEAGUES.values() if lg.extra}
            for (country, lname), grp in raw.groupby(["Country", "League"]):
                code = _extra_code(str(country), str(lname), lookup)
                if code:
                    frames.append(normalise_football_data(grp, code))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


_COUNTRY_TO_CODE = {
    "argentina": "ARG", "austria": "AUT", "brazil": "BRA", "china": "CHN",
    "denmark": "DEN", "finland": "FIN", "ireland": "IRL", "japan": "JAP",
    "mexico": "MEX", "norway": "NOR", "poland": "POL", "romania": "ROM",
    "russia": "RUS", "sweden": "SWE", "switzerland": "SUI", "usa": "USA",
}


def _extra_code(country: str, league_name: str, lookup: dict[str, str]) -> str | None:
    return _COUNTRY_TO_CODE.get(country.strip().lower()) or lookup.get(league_name.strip().lower())


def load_matches(raw_dir: Path, source: str = "auto", **kw) -> tuple[pd.DataFrame, str]:
    """Return (matches, source_used). ``source`` is auto|football-data|mirror."""
    if source in ("auto", "football-data"):
        try:
            return fetch_football_data(raw_dir, **kw), "football-data"
        except Exception as e:  # noqa: BLE001
            if source == "football-data":
                raise
            log.warning("football-data.co.uk unavailable (%s); using GitHub mirror", e)
    return fetch_mirror(raw_dir), "mirror"
