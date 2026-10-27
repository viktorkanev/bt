import pandas as pd

from footy.data.schema import normalise_football_data, normalise_mirror


def test_football_data_main_format():
    raw = pd.DataFrame({
        "Div": ["E1", "E1"], "Date": ["09/08/25", "16/08/2025"], "Time": ["15:00", "12:30"],
        "HomeTeam": ["Leeds", "Hull"], "AwayTeam": ["Hull", "Leeds"],
        "FTHG": [2, 0], "FTAG": [1, 0], "FTR": ["H", "D"], "HS": [12, 9], "AS": [7, 10],
        "HST": [5, 3], "AST": [2, 4], "B365H": [1.8, 2.5], "B365D": [3.6, 3.2], "B365A": [4.5, 2.9],
        "PSCH": [1.75, 2.6], "PSCD": [3.7, 3.25], "PSCA": [4.8, 2.95], "AHh": [-0.75, 0.25],
        "B365AHH": [1.95, 1.9], "B365AHA": [1.95, 2.0], "B365>2.5": [1.9, 2.1],
        "B365<2.5": [1.9, 1.75],
    })
    d = normalise_football_data(raw, "E1")
    assert list(d["date"].dt.day) == [9, 16]
    assert list(d["ftr"]) == ["H", "D"]
    assert d.loc[0, "psc_h"] == 1.75 and d.loc[0, "b365_o25"] == 1.9
    assert d.loc[0, "ah_line"] == -0.75
    assert d.loc[0, "season"] == "2025-26" and d.loc[0, "tier"] == 2


def test_extra_league_format_uses_closing_as_prematch():
    raw = pd.DataFrame({
        "Country": ["Brazil"], "League": ["Serie A"], "Season": [2024], "Date": ["13/04/2024"],
        "Time": ["20:00"], "Home": ["Flamengo"], "Away": ["Santos"], "HG": [3], "AG": [1],
        "Res": ["H"], "PSCH": [1.5], "PSCD": [4.2], "PSCA": [6.5], "B365CH": [1.5],
        "B365CD": [4.0], "B365CA": [6.0], "MaxCH": [1.55], "MaxCD": [4.4], "MaxCA": [7.0],
    })
    d = normalise_football_data(raw, "BRA")
    assert d.loc[0, "season"] == "2024"
    assert d.loc[0, "b365_h"] == 1.5 and d.loc[0, "max_a"] == 7.0


def test_mirror_rounding_of_handicap_lines_is_fixed():
    raw = pd.DataFrame({
        "Division": ["E0"], "MatchDate": ["2024-08-17"], "MatchTime": ["15:00:00"],
        "HomeTeam": ["Arsenal"], "AwayTeam": ["Wolves"], "FTHome": [2], "FTAway": [0],
        "FTResult": ["H"], "OddHome": [1.2], "OddDraw": [7.0], "OddAway": [13.0],
        "HandiSize": [-0.8], "HandiHome": [1.9], "HandiAway": [2.0],
    })
    d = normalise_mirror(raw)
    assert d.loc[0, "ah_line"] == -0.75
