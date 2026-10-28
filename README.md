# footy: football rating, prediction and backtesting engine

Tracks results across 38 divisions (England down to the National League,
Scotland down to League Two, second tiers in Germany, Italy, Spain and France,
16 other top flights), rates every team, predicts match probabilities, compares
them with bookmaker prices and backtests betting strategies without look-ahead.

**Open the report:** `reports/latest/index.html` (self-contained, works offline).
**Research notes:** [`docs/RESEARCH.md`](docs/RESEARCH.md): metrics, data sources,
team-news accounts, literature and findings.

## Headline results (holdout seasons 2016/17 to Sep 2026)

* Bookmaker prices (Bet365, margin removed) score RPS **0.2039**; the best
  ratings-only model **0.2083**. Ratings alone lose money at every price tested.
* 1X2 "best price" profits mostly come from stale or outlier quotes; with
  edges above 5% ignored they shrink to about zero.
* **Over/Under 2.5 in divisions 2-5** (model + market, best price, edge >= 3%):
  **+7.5% ROI over 3,061 holdout bets** (95% range +3.8% to +11.1%), +3.0% in
  development, positive in all 11 lower-tier European leagues. Top flights:
  -0.8%. This is a lead to paper-trade, not a proven edge (see
  `docs/RESEARCH.md` section 7).

## New: does the market confuse luck with skill? (pre-registered)

`docs/PREREGISTRATION.md` and `docs/SURPRISE_RESULTS.md`. Each team's recent
performance against the bookmaker's own expectation is split into results,
shot dominance, finishing luck and goalkeeper luck. Rules were committed to git
before a one-shot holdout test on 2016-2026:

* **Confirmed:** bookmakers over-react to finishing and goalkeeper luck
  (holdout t = -5.2 and -4.7 over 68,270 matches, p = 1.7e-16).
* **Not profitable at Bet365:** the three pre-registered betting rules failed
  (best: Asian handicap +1.2% over 6,034 bets, not significant). Re-priced at a
  2% margin the same bets project to +2.5% (development and holdout alike).
* **Forward test:** `footy forward` scores two frozen rules on matches after
  2026-09-03 with closing-line value (Pinnacle, then Betfair Exchange closing).
* **Against closing prices** (`docs/CLV_RESULTS.md`, `footy clv`): the luck bias
  survives even Pinnacle's closing line (t = -4.3, -3.8) but is worth about 1% at
  sharp prices. The lower-division over/under rule beats the close (+1.2% CLV,
  +4.8% ROI over 3,660 bets) only at the best price across bookmakers: a
  line-shopping edge.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dashboard,dev]"

footy all                 # fetch data -> ratings + walk-forward models -> report
streamlit run app/dashboard.py
```

`footy fetch` downloads from football-data.co.uk (with Pinnacle and closing
odds, plus upcoming fixtures). If that site is unreachable it falls back to a
GitHub mirror of the same data (no closing odds, no fixtures).

### Commands

| Command | What it does |
|---|---|
| `footy fetch [--source auto\|football-data\|mirror] [--leagues E1,E2]` | Download results, stats, odds and upcoming fixtures |
| `footy build [--no-gbm]` | Run all rating models over history, train walk-forward models, save predictions (about 8 minutes for 240k matches) |
| `footy report` | Accuracy tables, strategy grid, dev/holdout backtests, HTML report in `reports/latest/` |
| `footy backtest --market ou25 --model oublend --price max --edge 0.03 --leagues E1,E2,E3` | Backtest one rule and print the summary plus per-league table |
| `footy upcoming [--no-fetch]` | Refresh this season's results, pull fixtures with odds, score them (market, research-adjusted and model probabilities, F1/F2 picks), update the paper-trading ledger. Writes `reports/upcoming/index.html` |
| `footy predict` | Same as `footy upcoming --no-fetch` |
| `footy matchup E0 Arsenal Chelsea` | Predict any match-up from the latest ratings |
| `footy tune` | Re-tune rating hyper-parameters on the 2003-2013 development window |
| `footy confirm [--window dev\|holdout]` | Re-run the pre-registered market-surprise test |
| `footy clv` | Closing-line value of the frozen rules on 2016-2026 (needs football-data.co.uk data with closing odds) |
| `footy forward` | Score the frozen rules (`config/forward_rules.json`) on matches after 2026-09-03 |

### Weekly routine

football-data.co.uk publishes fixtures with odds around **Tuesday** (midweek games)
and **Friday** (weekend games). Run `footy upcoming` after each update:

1. it refreshes this season's results and settles earlier paper trades;
2. it scores the new fixtures and logs new F1/F2 picks with the price available now
   (`reports/research/forward/ledger.csv`, first price seen is kept);
3. open `reports/upcoming/index.html` or the dashboard's *Upcoming & value* page.

Rebuild the models (`footy build`, ~10 min) once a month or so; ratings themselves are
recomputed from the latest results on every `footy upcoming` run.

To run it automatically on Windows (Task Scheduler, Tuesdays and Fridays at 18:00):

```bat
schtasks /Create /TN "footy upcoming" /SC WEEKLY /D TUE,FRI /ST 18:00 ^
  /TR "cmd /c cd /d C:\path\to\bt && footy upcoming > reports\upcoming\last_run.log 2>&1"
```

### Dashboard pages

* **Backtest lab**: choose market (1X2, O/U 2.5, Asian handicap), model, price,
  edge range, odds range, tiers/leagues, flat or Kelly staking; get the equity
  curve, ROI with confidence range, p-value, per-league and per-season tables,
  and a stale-price check.
* **Model accuracy**: RPS by model and league, calibration plots.
* **Match predictor**: any two teams, all models, scoreline grid, O/U and
  handicap fair odds, team-news sliders.
* **Upcoming & value**: market vs research-adjusted vs model probabilities for every
  fixture, the F1/F2 research picks, and the paper-trading ledger (ROI and closing-line
  value). Has a button to refresh results and fixtures.
* **Ratings** and **Team news** (manual adjustments from `config/adjustments.yaml`).

## How it works

```
data/            football-data.co.uk or mirror -> one schema (src/footy/data)
   |
ratings engine   day-by-day, pre-match only (src/footy/ratings)
   |             Elo (self-adjusting home advantage), pi-ratings,
   |             online Dixon-Coles on goals and on pseudo-xG (goals + shots on target),
   |             shots / shots-on-target ratings, promotion & relegation re-basing
   |
walk-forward     per season, trained on earlier seasons only (src/footy/models)
   |             ordered logit, logistic regression, gradient boosting,
   |             model + market blend, over/under models; margin removal (Shin/power)
   |
backtester       value bets, flat or fractional Kelly, CLV, bootstrap ranges,
   |             dev window picks / holdout judges (src/footy/backtest.py)
   |
report + app     reports/latest/index.html, app/dashboard.py
```

Key design rules:

* **No look-ahead.** Features for day D use only matches before D (tested in
  `tests/test_engine.py`). Probability models are re-fitted each season on
  earlier seasons.
* **Pick on development, judge on holdout.** Hyper-parameters were tuned on
  2003-2013; strategy rules are scored on 2005-2016 and reported on 2016-2026.
* **Report the uncomfortable numbers.** ROI by claimed edge, capped-edge
  variants and a dev-vs-holdout scatter of every strategy variant are in the
  report.

## Team news

Edit `config/adjustments.yaml`:

```yaml
- team: Leeds
  country: ENG
  attack: -0.08      # first-choice striker out
  defence: 0.0
  until: 2026-11-09
  note: "Striker out, per Thursday presser"
```

Sources for news (X accounts, injury sites, lineup timings) are listed in
`docs/RESEARCH.md` section 5.

## Repository layout

```
src/footy/
  data/        sources.py (downloaders), schema.py (normalisation), store.py
  ratings/     models.py (Elo, pi, online Poisson), engine.py, poisson_math.py
  models/      market.py (de-vig), probability.py (walk-forward models)
  backtest.py  strategies, staking, metrics
  report.py    report tables;  report_html.py  static dashboard
  pipeline.py  build / predict / team-news adjustments
  cli.py       `footy` command
app/dashboard.py   Streamlit app
config/            adjustments.yaml
docs/RESEARCH.md   research notes
reports/latest/    generated report (CSV tables + index.html)
tests/             pytest suite
```

## Limitations

* The included report was built from the GitHub mirror (the build environment
  could not reach football-data.co.uk): no Pinnacle or closing odds, so no CLV,
  and no upcoming fixtures. Run `footy all` locally to get both.
* "Best price" assumes you can always get the top quote. You often cannot,
  and bookmakers limit winning accounts, especially in lower leagues.
* Only league matches are in the data; cup and European games are missing from
  rest-day and form calculations.

Gambling involves financial risk. Nothing here guarantees future returns.
