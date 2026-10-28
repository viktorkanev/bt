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
    """Parse a football-data.co.uk CSV.

    Files are mostly UTF-8 with a byte-order mark (BOM), older ones latin-1.
    Decoding a BOM as latin-1 mangles the first column name ("Div"), which made
    fixture files unreadable, so try UTF-8 first and clean the header either way.
    """
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    df = pd.read_csv(io.StringIO(text), on_bad_lines="skip", low_memory=False)
    df.columns = [str(c).replace("\ufeff", "").replace("\u00ef\u00bb\u00bf", "").strip()
                  for c in df.columns]
    return df


def _looks_like_csv(content: bytes) -> bool:
    """False for HTML error pages served for seasons a league is not covered."""
    head = content[:400].replace(b"\xef\xbb\xbf", b"").lstrip().lower()
    return bool(head) and not head.startswith(b"<") and b"date" in head


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
                    content = _get(url)
                except Exception as e:  # noqa: BLE001
                    log.info("skip %s: %s", url, e)
                    continue
                if not _looks_like_csv(content):
                    log.info("skip %s: not a results file (league not covered that season)", url)
                    continue
                path.write_bytes(content)
            content = path.read_bytes()
            if not _looks_like_csv(content):
                path.unlink(missing_ok=True)  # error page cached by an older version
                continue
            try:
                frames.append(normalise_football_data(_read_csv_bytes(content), code))
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


def parse_fixtures(name: str, content: bytes) -> list[pd.DataFrame]:
    """Normalise fixtures.csv (main leagues) or new_league_fixtures.csv (extra leagues)."""
    raw = _read_csv_bytes(content)
    frames: list[pd.DataFrame] = []
    if "Div" in raw.columns:
        for div, grp in raw.dropna(subset=["Div"]).groupby("Div"):
            frames.append(normalise_football_data(grp, str(div).strip()))
    elif "Country" in raw.columns and "League" in raw.columns:
        lookup = {lg.name.lower(): lg.code for lg in leagues.LEAGUES.values() if lg.extra}
        for (country, lname), grp in raw.groupby(["Country", "League"]):
            code = _extra_code(str(country), str(lname), lookup)
            if code:
                frames.append(normalise_football_data(grp, code))
    if not frames:
        log.warning("%s: no fixtures recognised (%d rows, columns %s)", name, len(raw),
                    list(raw.columns)[:8])
    return frames


def fetch_fixtures(raw_dir: Path) -> pd.DataFrame:
    """Upcoming fixtures with current odds from football-data.co.uk."""
    raw_dir = Path(raw_dir) / "football-data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for name in ("fixtures.csv", "new_league_fixtures.csv"):
        try:
            content = _get(f"{FD_BASE}/{name}")
        except Exception as e:  # noqa: BLE001
            log.warning("fixtures download failed (%s): %s", name, e)
            continue
        (raw_dir / name).write_bytes(content)
        frames += parse_fixtures(name, content)
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


def unmapped_columns(raw_dir: Path | None = None, sample: tuple = ("E1", "E2", "SP2", "D2")) -> list[str]:
    """Columns in this season's football-data.co.uk files that the schema ignores.

    Used to spot new bookmakers (e.g. a Betfair exchange price) or renamed columns.
    """
    from .schema import FD_MAP
    from .store import RAW_DIR

    raw_dir = Path(raw_dir or RAW_DIR) / "football-data"
    known = {c for cands in FD_MAP.values() for c in cands}
    known |= {"Div", "Date", "Time", "HomeTeam", "AwayTeam", "Home", "Away", "HT", "AT", "Referee",
              "Country", "League", "Season"}
    seen: set[str] = set()
    code = fd_season_code(current_season_start())
    for lg in sample:
        f = raw_dir / f"{lg}_{code}.csv"
        if f.exists():
            seen |= set(_read_csv_bytes(f.read_bytes()).columns)
    return sorted(c for c in seen - known if c and not c.startswith("Unnamed"))


def refresh_current_season(matches: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """Re-download only the current season (and the extra-league files) and merge.

    Much faster than a full ``fetch`` and enough to bring ratings up to date
    before predicting fixtures. Only valid for a store built from football-data.co.uk
    (the mirror uses different team names).
    """
    cur = current_season_start()
    new = fetch_football_data(raw_dir, first_season=cur, refresh_recent=1)
    extra = {lg.code for lg in leagues.LEAGUES.values() if lg.extra}
    replaced = new["league"].unique()
    drop = matches["league"].isin(replaced) & (
        matches["league"].isin(extra) | (matches["season_start"] >= cur))
    merged = pd.concat([matches[~drop], new], ignore_index=True)
    return finalise(merged)


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
