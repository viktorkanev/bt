"""Watcher configuration: TOML (Python 3.11+) or JSON, merged over defaults.

Secrets can come from environment variables instead of the file:
FOOTY_ODDSAPI_KEY, FOOTY_BETFAIR_APP_KEY, FOOTY_BETFAIR_USERNAME, FOOTY_BETFAIR_PASSWORD,
FOOTY_TELEGRAM_TOKEN, FOOTY_TELEGRAM_CHAT_ID, FOOTY_NTFY_TOPIC, FOOTY_WATCH_TOKEN.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

# football-data.co.uk league code for each the-odds-api sport key. Keys marked
# "verify" may not exist on your plan; run `python -m footy.watch sports`.
ODDSAPI_SPORTS = {
    "soccer_epl": "E0", "soccer_efl_champ": "E1", "soccer_england_league1": "E2",
    "soccer_england_league2": "E3", "soccer_spl": "SC0",
    "soccer_germany_bundesliga": "D1", "soccer_germany_bundesliga2": "D2",
    "soccer_italy_serie_a": "I1", "soccer_italy_serie_b": "I2",
    "soccer_spain_la_liga": "SP1", "soccer_spain_segunda_division": "SP2",
    "soccer_france_ligue_one": "F1", "soccer_france_ligue_two": "F2",
    "soccer_netherlands_eredivisie": "N1", "soccer_portugal_primeira_liga": "P1",
    "soccer_belgium_first_div": "B1", "soccer_turkey_super_league": "T1",
    "soccer_greece_super_league": "G1",
}

DEFAULTS: dict = {
    "server": {"host": "0.0.0.0", "port": 8080, "token": "", "db": "footy-watch.db",
               "picks_file": "picks.json", "timezone": "Europe/London", "loop_seconds": 5},
    "alerts": {
        "markets": ["1x2", "ou", "ah"],
        "reference_books": ["pinnacle", "betfair_ex_eu", "betfair"],
        "target_books": ["*"],          # books you can bet at; "*" = every non-reference book
        "pick_books": ["*"],
        "min_edge": 0.03, "max_edge": 0.25,
        "pick_extra_edge": 0.0,         # extra margin of safety on research picks
        "lookahead_hours": 36, "min_minutes_before_kickoff": 2,
        "max_quote_age_minutes": 30, "max_exchange_spread": 0.05, "min_exchange_size": 20,
        "cooldown_minutes": 240, "reprice_step": 0.01, "value_alerts": True, "pick_alerts": True,
    },
    "commission": {"betfair_ex_eu": 0.05, "betfair_ex_uk": 0.05, "betfair_ex_au": 0.05,
                   "betfair": 0.05, "matchbook": 0.02, "smarkets": 0.02},
    "oddsapi": {"enabled": False, "api_key": "", "regions": "eu", "markets": ["h2h", "totals", "spreads"],
                "bookmakers": [], "monthly_credits": 500, "min_interval_seconds": 600,
                "lookahead_hours": 36, "events_refresh_minutes": 60, "sports": ODDSAPI_SPORTS},
    "betfair": {"enabled": False, "app_key": "", "username": "", "password": "", "interval_seconds": 30,
                "lookahead_hours": 36, "competitions": [], "catalogue_refresh_minutes": 15},
    "file": {"enabled": False, "path": "odds.json", "interval_seconds": 30},
    "notify": {"console": True, "telegram_token": "", "telegram_chat_id": "", "ntfy_topic": "",
               "ntfy_server": "https://ntfy.sh"},
}

ENV = {
    ("oddsapi", "api_key"): "FOOTY_ODDSAPI_KEY", ("betfair", "app_key"): "FOOTY_BETFAIR_APP_KEY",
    ("betfair", "username"): "FOOTY_BETFAIR_USERNAME", ("betfair", "password"): "FOOTY_BETFAIR_PASSWORD",
    ("notify", "telegram_token"): "FOOTY_TELEGRAM_TOKEN", ("notify", "telegram_chat_id"): "FOOTY_TELEGRAM_CHAT_ID",
    ("notify", "ntfy_topic"): "FOOTY_NTFY_TOPIC", ("server", "token"): "FOOTY_WATCH_TOKEN",
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "sports":
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: str | None) -> dict:
    data: dict = {}
    if path:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix == ".json":
            data = json.loads(text)
        else:
            import tomllib  # Python 3.11+
            data = tomllib.loads(text)
    cfg = _merge(DEFAULTS, data)
    for (sec, key), var in ENV.items():
        if os.environ.get(var):
            cfg[sec][key] = os.environ[var]
    base = Path(path).resolve().parent if path else Path.cwd()
    for key in ("db", "picks_file"):
        if not Path(cfg["server"][key]).is_absolute():
            cfg["server"][key] = str(base / cfg["server"][key])
    if cfg["file"].get("path") and not Path(cfg["file"]["path"]).is_absolute():
        cfg["file"]["path"] = str(base / cfg["file"]["path"])
    return cfg
