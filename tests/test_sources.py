import ast
from pathlib import Path

from footy.data import sources
from footy.data.schema import normalise_football_data

FIXTURES = ("Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,MaxH,MaxD,MaxA,AHh,B365AHH,B365AHA,"
            "PAHH,PAHA,B365>2.5,B365<2.5\n"
            "E0,01/11/2026,15:00,Arsenal,Chelsea,2.10,3.40,3.50,2.15,3.5,3.6,-0.25,1.95,1.95,1.97,1.95,1.8,2.0\n"
            "E2,01/11/2026,15:00,Bolton,Wigan,1.90,3.50,4.00,1.95,3.6,4.2,-0.5,1.9,2.0,1.92,1.98,1.9,1.9\n")


def test_fixtures_with_utf8_bom_are_recognised():
    """football-data.co.uk serves fixtures.csv with a BOM; it used to hide the Div column."""
    frames = sources.parse_fixtures("fixtures.csv", ("﻿" + FIXTURES).encode("utf-8"))
    assert sorted(f["league"].iloc[0] for f in frames) == ["E0", "E2"]
    e0 = [f for f in frames if f["league"].iloc[0] == "E0"][0]
    assert e0.loc[e0.index[0], "ah_line"] == -0.25 and e0["ftr"].isna().all()


def test_latin1_file_parses():
    raw = ("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nSP1,20/08/2005,Málaga,Sevilla,1,1,D\n").encode("latin-1")
    df = normalise_football_data(sources._read_csv_bytes(raw), "SP1")
    assert df.loc[0, "home"] == "Málaga"


def test_old_ht_at_column_names():
    raw = b"Div,Date,HT,AT,FTHG,FTAG,FTR\nG1,26/08/01,AEK,PAOK,2,0,H\n"
    df = normalise_football_data(sources._read_csv_bytes(raw), "G1")
    assert (df.loc[0, "home"], df.loc[0, "away"], df.loc[0, "ftr"]) == ("AEK", "PAOK", "H")


def test_error_pages_are_not_treated_as_data():
    assert not sources._looks_like_csv(b"<!DOCTYPE html><html><body>Not found</body></html>")
    assert not sources._looks_like_csv(b"")
    assert sources._looks_like_csv(("﻿" + FIXTURES).encode("utf-8"))


def test_no_platform_default_encoding():
    """Text I/O must name its encoding: Windows defaults to cp1252 and crashed on the report."""
    root = Path(__file__).resolve().parents[1]
    bad = []
    for path in list((root / "src").rglob("*.py")) + list((root / "app").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            kws = {k.arg for k in node.keywords}
            if name in ("read_text", "write_text") and "encoding" not in kws:
                bad.append(f"{path.name}:{node.lineno}")
            if name == "open" and "encoding" not in kws:
                mode = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else "r"
                if "b" not in str(mode):
                    bad.append(f"{path.name}:{node.lineno}")
    assert not bad, bad
