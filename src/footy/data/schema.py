"""Canonical match schema and normalisers for each raw source.

Every loader returns a DataFrame with (a subset of) these columns. Odds columns
are named ``{book}_{outcome}`` where book is one of

* ``b365``  Bet365 pre-match odds (collected ~2 days before kick-off)
* ``ps``    Pinnacle pre-match odds
* ``max``   best price across the market
* ``avg``   market average
* ``bfe``   Betfair Exchange back price (no margin, but commission on winnings)
* the same with a ``c`` suffix (``b365c``, ``psc``, ``maxc``, ``avgc``) = closing odds

and outcome is ``h``/``d``/``a`` (1X2), ``o25``/``u25`` (over/under 2.5 goals)
or ``ahh``/``aha`` (Asian handicap home/away at line ``ah_line``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import leagues

BASE_COLS = [
    "match_id", "league", "country", "tier", "season", "season_start", "date", "time",
    "home", "away", "fthg", "ftag", "ftr", "hthg", "htag",
    "hs", "as_", "hst", "ast", "hc", "ac", "hf", "af", "hy", "ay", "hr", "ar",
]

BOOKS = ["b365", "ps", "max", "avg", "bfe", "b365c", "psc", "maxc", "avgc", "bfec"]
ODDS_1X2 = [f"{b}_{o}" for b in BOOKS for o in ("h", "d", "a")]
ODDS_OU = [f"{b}_{o}" for b in BOOKS for o in ("o25", "u25")]
ODDS_AH = ["ah_line", "ahc_line"] + [f"{b}_{o}" for b in BOOKS for o in ("ahh", "aha")]
ODDS_COLS = ODDS_1X2 + ODDS_OU + ODDS_AH
XG_COLS = ["hxg", "axg"]  # expected goals, in football-data.co.uk files from 2026-27
ALL_COLS = BASE_COLS + XG_COLS + ODDS_COLS

# football-data.co.uk column candidates (first non-null wins).
FD_MAP: dict[str, list[str]] = {
    "fthg": ["FTHG", "HG"],
    "ftag": ["FTAG", "AG"],
    "ftr": ["FTR", "Res"],
    "hthg": ["HTHG"],
    "htag": ["HTAG"],
    "hs": ["HS"], "as_": ["AS"], "hst": ["HST"], "ast": ["AST"],
    "hc": ["HC"], "ac": ["AC"], "hf": ["HF"], "af": ["AF"],
    "hy": ["HY"], "ay": ["AY"], "hr": ["HR"], "ar": ["AR"],
    "hxg": ["HxG"], "axg": ["AxG"],
    "bfe_h": ["BFEH"], "bfe_d": ["BFED"], "bfe_a": ["BFEA"],
    "bfec_h": ["BFECH"], "bfec_d": ["BFECD"], "bfec_a": ["BFECA"],
    "bfe_o25": ["BFE>2.5"], "bfe_u25": ["BFE<2.5"], "bfec_o25": ["BFEC>2.5"], "bfec_u25": ["BFEC<2.5"],
    "bfe_ahh": ["BFEAHH"], "bfe_aha": ["BFEAHA"], "bfec_ahh": ["BFECAHH"], "bfec_aha": ["BFECAHA"],
    "b365_h": ["B365H"], "b365_d": ["B365D"], "b365_a": ["B365A"],
    "ps_h": ["PSH", "PH"], "ps_d": ["PSD", "PD"], "ps_a": ["PSA", "PA"],
    "max_h": ["MaxH", "BbMxH"], "max_d": ["MaxD", "BbMxD"], "max_a": ["MaxA", "BbMxA"],
    "avg_h": ["AvgH", "BbAvH"], "avg_d": ["AvgD", "BbAvD"], "avg_a": ["AvgA", "BbAvA"],
    "b365c_h": ["B365CH"], "b365c_d": ["B365CD"], "b365c_a": ["B365CA"],
    "psc_h": ["PSCH"], "psc_d": ["PSCD"], "psc_a": ["PSCA"],
    "maxc_h": ["MaxCH"], "maxc_d": ["MaxCD"], "maxc_a": ["MaxCA"],
    "avgc_h": ["AvgCH"], "avgc_d": ["AvgCD"], "avgc_a": ["AvgCA"],
    "b365_o25": ["B365>2.5"], "b365_u25": ["B365<2.5"],
    "ps_o25": ["P>2.5"], "ps_u25": ["P<2.5"],
    "max_o25": ["Max>2.5", "BbMx>2.5"], "max_u25": ["Max<2.5", "BbMx<2.5"],
    "avg_o25": ["Avg>2.5", "BbAv>2.5"], "avg_u25": ["Avg<2.5", "BbAv<2.5"],
    "b365c_o25": ["B365C>2.5"], "b365c_u25": ["B365C<2.5"],
    "psc_o25": ["PC>2.5"], "psc_u25": ["PC<2.5"],
    "maxc_o25": ["MaxC>2.5"], "maxc_u25": ["MaxC<2.5"],
    "avgc_o25": ["AvgC>2.5"], "avgc_u25": ["AvgC<2.5"],
    "ah_line": ["AHh", "BbAHh"], "ahc_line": ["AHCh"],
    "b365_ahh": ["B365AHH"], "b365_aha": ["B365AHA"],
    "ps_ahh": ["PAHH"], "ps_aha": ["PAHA"],
    "max_ahh": ["MaxAHH", "BbMxAHH"], "max_aha": ["MaxAHA", "BbMxAHA"],
    "avg_ahh": ["AvgAHH", "BbAvAHH"], "avg_aha": ["AvgAHA", "BbAvAHA"],
    "b365c_ahh": ["B365CAHH"], "b365c_aha": ["B365CAHA"],
    "psc_ahh": ["PCAHH"], "psc_aha": ["PCAHA"],
    "maxc_ahh": ["MaxCAHH"], "maxc_aha": ["MaxCAHA"],
    "avgc_ahh": ["AvgCAHH"], "avgc_aha": ["AvgCAHA"],
}

# Mirror dataset (github.com/xgabora/Club-Football-Match-Data-2000-2025).
MIRROR_MAP: dict[str, str] = {
    "Division": "league", "MatchDate": "date", "MatchTime": "time",
    "HomeTeam": "home", "AwayTeam": "away",
    "FTHome": "fthg", "FTAway": "ftag", "FTResult": "ftr",
    "HTHome": "hthg", "HTAway": "htag",
    "HomeShots": "hs", "AwayShots": "as_", "HomeTarget": "hst", "AwayTarget": "ast",
    "HomeFouls": "hf", "AwayFouls": "af", "HomeCorners": "hc", "AwayCorners": "ac",
    "HomeYellow": "hy", "AwayYellow": "ay", "HomeRed": "hr", "AwayRed": "ar",
    "OddHome": "b365_h", "OddDraw": "b365_d", "OddAway": "b365_a",
    "MaxHome": "max_h", "MaxDraw": "max_d", "MaxAway": "max_a",
    "Over25": "b365_o25", "Under25": "b365_u25",
    "MaxOver25": "max_o25", "MaxUnder25": "max_u25",
    "HandiSize": "ah_line", "HandiHome": "b365_ahh", "HandiAway": "b365_aha",
}


def _coalesce(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for c in candidates:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            out = out.fillna(vals)
    return out


def normalise_football_data(raw: pd.DataFrame, league: str) -> pd.DataFrame:
    """Normalise one football-data.co.uk CSV (main or extra-league format)."""
    raw = raw.copy()
    raw.columns = [str(c).replace("\ufeff", "").replace("\u00ef\u00bb\u00bf", "").strip()
                   for c in raw.columns]
    # Column names changed over the years: HomeTeam/AwayTeam, Home/Away, and HT/AT
    # in some files from the early 2000s.
    home_col = next((c for c in ("HomeTeam", "Home", "HT") if c in raw.columns), None)
    away_col = next((c for c in ("AwayTeam", "Away", "AT") if c in raw.columns), None)
    if home_col is None or away_col is None or "Date" not in raw.columns:
        raise ValueError(f"not a match file (columns: {list(raw.columns)[:8]})")
    raw = raw.dropna(subset=[home_col, away_col, "Date"])
    out = pd.DataFrame(index=raw.index)
    out["league"] = league
    out["date"] = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed", errors="coerce")
    out["time"] = raw["Time"].astype(str) if "Time" in raw.columns else ""
    out["home"] = raw[home_col].astype(str).str.strip()
    out["away"] = raw[away_col].astype(str).str.strip()
    for col, cands in FD_MAP.items():
        if col == "ftr":
            src = next((c for c in cands if c in raw.columns), None)
            out[col] = raw[src].astype(str).str.strip() if src else np.nan
        else:
            out[col] = _coalesce(raw, cands)
    lg = leagues.get(league)
    if lg.extra:
        # Extra-league files only carry closing odds; expose them as the
        # pre-match columns too so every league has a usable "b365"/"max" price.
        for book in ("b365", "max", "avg", "ps"):
            for o in ("h", "d", "a"):
                out[f"{book}_{o}"] = out[f"{book}_{o}"].fillna(out[f"{book}c_{o}"])
    return finalise(out)


def normalise_mirror(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.rename(columns=MIRROR_MAP)
    out = out[[c for c in MIRROR_MAP.values() if c in out.columns]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["time"] = out["time"].fillna("").astype(str)
    for c in out.columns:
        if c not in ("league", "date", "time", "home", "away", "ftr"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # ClubElo snapshots are kept for reference only (not used by the models).
    if "HomeElo" in raw.columns:
        out["clubelo_h"] = pd.to_numeric(raw["HomeElo"], errors="coerce")
        out["clubelo_a"] = pd.to_numeric(raw["AwayElo"], errors="coerce")
    return finalise(out)


def finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns, drop junk rows, enforce column order and dtypes."""
    df = df.dropna(subset=["date", "home", "away"]).copy()
    df = df[(df["home"] != "") & (df["away"] != "") & (df["home"] != "nan")]
    for c in ALL_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df["ftr"] = df["ftr"].astype(object)
    meta = {code: leagues.get(code) for code in df["league"].unique()}
    df["country"] = df["league"].map(lambda c: meta[c].country)
    df["tier"] = df["league"].map(lambda c: meta[c].tier).astype(int)
    df["season_start"] = [
        leagues.season_start_year(c, d) for c, d in zip(df["league"], df["date"])
    ]
    df["season"] = [leagues.season_label(c, y) for c, y in zip(df["league"], df["season_start"])]
    # Result from goals when present (more reliable than the FTR column).
    played = df["fthg"].notna() & df["ftag"].notna()
    df.loc[played, "ftr"] = np.select(
        [df.loc[played, "fthg"] > df.loc[played, "ftag"],
         df.loc[played, "fthg"] < df.loc[played, "ftag"]],
        ["H", "A"], default="D",
    )
    df.loc[~played, "ftr"] = np.nan
    # Handicap lines come in 0.25 steps; some sources round them (-0.25 -> -0.3).
    for c in ("ah_line", "ahc_line"):
        df[c] = np.round(df[c] * 4) / 4
    # Odds <= 1 are data errors.
    for c in ODDS_COLS:
        if c not in ("ah_line", "ahc_line"):
            df.loc[df[c] <= 1.0, c] = np.nan
    df["time"] = df["time"].replace({"nan": "", "None": ""}).fillna("")
    df["match_id"] = (
        df["league"] + "_" + df["date"].dt.strftime("%Y%m%d") + "_"
        + df["home"].str.replace(r"\W+", "", regex=True) + "_"
        + df["away"].str.replace(r"\W+", "", regex=True)
    )
    df = df.drop_duplicates("match_id", keep="last")
    extra = [c for c in df.columns if c not in ALL_COLS]
    df = df[ALL_COLS + extra].sort_values(["date", "league", "home"]).reset_index(drop=True)
    return df
