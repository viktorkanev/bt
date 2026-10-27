"""League metadata.

Codes follow football-data.co.uk for the main European leagues (E0, SP1, ...)
and three-letter codes for the "extra" leagues. Where football-data.co.uk uses a
different code for the extra-league CSV files, ``fd_code`` holds it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class League:
    code: str
    name: str
    country: str
    tier: int
    calendar_season: bool = False  # True when the season runs Jan-Dec
    extra: bool = False  # football-data.co.uk "extra leagues" format (new/XXX.csv)
    fd_code: str | None = None  # code used by football-data.co.uk if different


_LEAGUES = [
    # England
    League("E0", "Premier League", "ENG", 1),
    League("E1", "Championship", "ENG", 2),
    League("E2", "League One", "ENG", 3),
    League("E3", "League Two", "ENG", 4),
    League("EC", "National League", "ENG", 5),
    # Scotland
    League("SC0", "Scottish Premiership", "SCO", 1),
    League("SC1", "Scottish Championship", "SCO", 2),
    League("SC2", "Scottish League One", "SCO", 3),
    League("SC3", "Scottish League Two", "SCO", 4),
    # Germany / Italy / Spain / France
    League("D1", "Bundesliga", "GER", 1),
    League("D2", "2. Bundesliga", "GER", 2),
    League("I1", "Serie A", "ITA", 1),
    League("I2", "Serie B", "ITA", 2),
    League("SP1", "La Liga", "ESP", 1),
    League("SP2", "Segunda Division", "ESP", 2),
    League("F1", "Ligue 1", "FRA", 1),
    League("F2", "Ligue 2", "FRA", 2),
    # Other European top flights
    League("N1", "Eredivisie", "NED", 1),
    League("B1", "Jupiler Pro League", "BEL", 1),
    League("P1", "Primeira Liga", "POR", 1),
    League("T1", "Super Lig", "TUR", 1),
    League("G1", "Super League Greece", "GRE", 1),
    # Extra leagues (football-data.co.uk new/XXX.csv)
    League("ARG", "Argentina Primera", "ARG", 1, calendar_season=True, extra=True),
    League("AUT", "Austrian Bundesliga", "AUT", 1, extra=True),
    League("BRA", "Brasileirao Serie A", "BRA", 1, calendar_season=True, extra=True),
    League("CHN", "Chinese Super League", "CHN", 1, calendar_season=True, extra=True),
    League("DEN", "Danish Superliga", "DEN", 1, extra=True, fd_code="DNK"),
    League("FIN", "Veikkausliiga", "FIN", 1, calendar_season=True, extra=True),
    League("IRL", "League of Ireland", "IRL", 1, calendar_season=True, extra=True),
    League("JAP", "J1 League", "JAP", 1, calendar_season=True, extra=True, fd_code="JPN"),
    League("MEX", "Liga MX", "MEX", 1, extra=True),
    League("NOR", "Eliteserien", "NOR", 1, calendar_season=True, extra=True),
    League("POL", "Ekstraklasa", "POL", 1, extra=True),
    League("ROM", "Liga I", "ROM", 1, extra=True, fd_code="ROU"),
    League("RUS", "Russian Premier League", "RUS", 1, extra=True),
    League("SWE", "Allsvenskan", "SWE", 1, calendar_season=True, extra=True),
    League("SUI", "Swiss Super League", "SUI", 1, extra=True, fd_code="SWZ"),
    League("USA", "MLS", "USA", 1, calendar_season=True, extra=True),
]

LEAGUES: dict[str, League] = {lg.code: lg for lg in _LEAGUES}
FD_EXTRA_CODES: dict[str, str] = {(lg.fd_code or lg.code): lg.code for lg in _LEAGUES if lg.extra}


def get(code: str) -> League:
    if code not in LEAGUES:
        # Unknown league: treat as a European-calendar top flight of its own country.
        return League(code, code, code, 1)
    return LEAGUES[code]


def league_name(code: str) -> str:
    return get(code).name


def season_start_year(code: str, date) -> int:
    """Season start year for a match date (July cut-off unless calendar season)."""
    lg = get(code)
    if lg.calendar_season:
        return date.year
    return date.year if date.month >= 7 else date.year - 1


def season_label(code: str, start_year: int) -> str:
    if get(code).calendar_season:
        return str(start_year)
    return f"{start_year}-{str(start_year + 1)[-2:]}"
