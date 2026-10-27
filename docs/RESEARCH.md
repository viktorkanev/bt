# Research notes: predicting football results and finding an edge

This file covers what the engine uses and why: which metrics carry signal, where
data comes from, which team-news sources are worth following, what the
literature says, and what our own backtests found (October 2026 build, 238,854
matches, 38 divisions, 2000-2026).

---

> **Update:** a second, pre-registered study ("does the market confuse luck with
> skill?") is in `docs/PREREGISTRATION.md` and `docs/SURPRISE_RESULTS.md`.

## 1. Findings from our backtests

All numbers are out-of-sample. Models are re-trained each season on earlier data
only (walk-forward). Hyper-parameters were tuned on 2003-2013. Strategy rules
were fixed on the **development window (Jul 2005 - Jun 2016)** and judged on the
**holdout (Jul 2016 - Sep 2026)**.

### 1.1 Accuracy: the market is very hard to beat on 1X2

Holdout, 110,779 matches. RPS = ranked probability score (lower is better).

| Model | RPS | Log loss | Accuracy |
|---|---|---|---|
| Bet365, margin removed (power method) | **0.2039** | 1.0011 | 50.4% |
| Ensemble + market (blend) | 0.2039 | 1.0011 | 50.4% |
| Ensemble, ratings only (logistic regression) | 0.2083 | 1.0153 | 49.4% |
| Gradient boosting, ratings only | 0.2084 | 1.0160 | 49.3% |
| Dixon-Coles on pseudo-xG (goals + shots on target) | 0.2093 | 1.0188 | 49.2% |
| Dixon-Coles on goals | 0.2096 | 1.0194 | 49.1% |
| Elo | 0.2097 | 1.0200 | 49.1% |
| Pi-ratings | 0.2098 | 1.0205 | 49.0% |

* A gap of about 0.004 RPS between the best public-data model and the bookmaker
  is in line with the literature.
* Feeding the bookmaker's own probabilities into the model ("blend") gives back
  the market almost exactly: on average the ratings add nothing the price does
  not already contain.
* Shot data helps: blending shots on target into the goals model (pseudo-xG)
  beats goals alone in every period.

### 1.2 Lower divisions vs top flights (European leagues, holdout)

| | Top flights | Second tiers | Tiers 3-5 |
|---|---|---|---|
| Bet365 margin (1X2) | 5.7% | 6.7% | 7.2% |
| Best-price margin (max across ~17 books) | 0.9% | 1.7% | 2.3% |
| Ratings-only gap to market (RPS x 10^-3) | 3.9 | 2.7 | 3.5 |

The market is relatively less sharp below the top flight (the model gets closer
to it), but margins are higher and best prices are worse. For 1X2 that roughly
cancels out.

### 1.3 Strategies (flat 1-unit stakes)

| Strategy (fixed in advance) | Dev bets | Dev ROI | Holdout bets | Holdout ROI | 95% range |
|---|---|---|---|---|---|
| Consensus: Bet365 fair price vs best price, edge >= 2% | 43,652 | +1.6% | 28,254 | +1.8% | -0.1% to +3.4% |
| Model+market @ best price, edge >= 3% | 54,928 | +3.0% | 44,935 | +1.8% | +0.4% to +3.2% |
| Model+market @ Bet365, edge >= 3% | 11,303 | -0.5% | 3,284 | -1.6% | -6.9% to +3.6% |
| Ratings-only @ best price, edge >= 5% | 64,823 | -0.8% | 79,390 | -3.1% | -4.2% to -2.1% |
| Over/Under 2.5 model+market @ best price, edge >= 3% | 20,609 | +2.9% | 6,477 | +3.1% | +0.8% to +5.7% |
| Asian handicap pseudo-xG @ Bet365, edge >= 5% | 31,320 | -3.7% | 34,899 | -4.1% | -5.0% to -3.3% |

What this means:

1. **Ratings alone lose money.** Even at the best price across the market a
   ratings-only model loses about 3% on the holdout.
2. **"Best price" 1X2 profits are mostly stale quotes.** The profit lives in bets
   where the top price is more than 8% above fair value. Those are usually a
   bookmaker that has not moved its line yet, or a data error. Ignoring edges
   above 5% cuts the model+market holdout ROI to +0.2%. In practice you rarely
   get those prices, and books that offer them limit winners quickly (the
   Kaunitz et al. paper below reports exactly that).
3. **Bet365 at its own prices cannot be beaten with public ratings** (-1.6%).
4. **Over/Under 2.5 in lower divisions is the one lead worth chasing.**

| O/U 2.5 model+market @ best price | Dev bets | Dev ROI | Holdout bets | Holdout ROI | Holdout 95% range |
|---|---|---|---|---|---|
| Top flights | 10,959 | +2.9% | 3,416 | -0.8% | -4.0% to +2.3% |
| Second tiers | 4,761 | +1.1% | 1,581 | +9.6% | +4.8% to +14.6% |
| Tiers 3-5 | 4,889 | +4.7% | 1,480 | +5.3% | -0.1% to +10.8% |
| **All tiers 2-5** | 9,650 | +3.0% | 3,061 | **+7.5%** | +3.8% to +11.1% |

On the holdout it was positive in all 11 lower-tier European leagues (E1, E2,
E3, National League, SC1-SC3, D2, I2, SP2, F2); in development, 8 of 11. The
top-flight version faded (+2.9% to -0.8%), which fits the idea that the
top-flight totals market has become efficient while lower leagues lag.

**Caveat.** The tier split of this strategy was examined after seeing results,
and it bets at best price. Treat it as a hypothesis to paper-trade, not a proven
edge. Section 7 has the plan.

### 1.4 A bug worth remembering

The GitHub mirror rounds Asian-handicap lines to one decimal (-0.25 becomes
-0.3). Settling bets at -0.3 instead of -0.25 produced a fake +3.5% handicap
"edge" over 47,000 bets. After snapping lines to 0.25 steps, betting every
handicap blindly loses about 3.9% on each side, matching the margin, and the
handicap strategy loses 4.1%. Lesson: a large, smooth profit from a simple
model is a bug until proven otherwise. There is a regression test for this.

---

## 2. What predicts football results

Ordered roughly by how much signal each adds on top of the previous ones.

| Signal | Why it matters | In the engine |
|---|---|---|
| **Betting market prices** | Aggregate the knowledge of everyone betting, including team news. Closing prices from sharp books (Pinnacle, Betfair) are the best public forecast available. | `mkt` probabilities (Bet365 de-vigged with the power method), used in `blend` |
| **Team strength from results** | Elo, pi-ratings and Dixon-Coles attack/defence explain most of the predictable variance. | `elo`, `pi`, `pois` |
| **Shot-based strength (xG, shots on target)** | Goals are noisy: one match of goals says little, shots say more. Wheatcroft's GAP ratings found shots on target the most informative observed statistic. | `pxg` (goals + shots on target x league conversion), `sot`, `shots` ratings |
| **Home advantage, by league and over time** | Varies by country and collapsed during the 2020-21 empty-stadium period. Bookmakers were slow to adjust (Winkelmann et al. 2021; Fischer & Haucap 2021). | Elo home advantage learns online per league; Poisson home term per league |
| **Promotion and relegation** | A promoted side's rating from the division below is on a different scale. | Re-based on entry using how earlier movers settled |
| **Player availability** (injuries, suspensions, rotation, lineups) | The biggest short-term information shock. Markets move within minutes of credible news; the edge is in speed or in leagues the market watches less. | Manual adjustments (`config/adjustments.yaml`, dashboard sliders) |
| **Schedule** (rest days, travel, European midweeks) | Small but real; congestion matters more for squads without depth. | Rest days computed; only league games in the data, so treat as rough |
| **Manager change, motivation, weather, referees** | Mostly small or already priced. Referee matters for cards markets. | Not modelled |

### Metrics for judging models and bets

* **RPS** (ranked probability score): standard for 1X2 because it respects the
  ordering home > draw > away. Also log loss, Brier and calibration plots.
* **Model vs market gap**: RPS(model) - RPS(market). Below zero is rare.
* **ROI / yield**: profit per unit staked. Needs thousands of bets to separate
  skill from luck at typical odds; we report bootstrap ranges and one-sided
  p-values.
* **Closing line value (CLV)**: how your price compares with the final
  (closing) price, margin removed. The fastest honest signal of skill: a few
  hundred bets with +2% CLV is more convincing than a profitable month. The
  backtester computes CLV automatically when Pinnacle or average closing odds
  are present (football-data.co.uk direct download has them from 2012/13 for
  Pinnacle).
* **Multiple testing**: the report scores 75 strategy variants. Some will look good by
  chance. The report shows dev vs holdout ROI for all of them.

---

## 3. Models and the literature behind them

| Paper | What we took from it |
|---|---|
| Maher (1982); Dixon & Coles (1997), *Applied Statistics* 46(2) | Independent Poisson attack/defence model with a low-score correction (rho) and time decay. Implemented online (stochastic-gradient updates) so it runs over 250k matches in seconds. |
| Hvattum & Arntzen (2010), *IJF* | Goal-difference weighted Elo as a forecasting input. |
| Constantinou & Fenton (2013), [*JQAS*](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2012-0036/html) | Pi-ratings: separate home/away ratings, log-dampened goal-difference errors. |
| Hubacek, Sourek & Zelezny (2019), [*Machine Learning*](https://ida.fel.cvut.cz/papers/hubacek2019learning.html) | Won the 2017 Soccer Prediction Challenge with pi-ratings + PageRank features and gradient-boosted trees; test RPS 0.2063 ([survey](https://arxiv.org/pdf/2403.07669)). Basis for our `gbm` model. |
| Wheatcroft (2020), [GAP ratings](https://arxiv.org/pdf/2001.09097) | Ratings built on match statistics; shots on target most informative observed stat, and xG not automatically better than other stats. Basis for `sot`/`pxg`. |
| Strumbelj (2014), *IJF* | Shin's method beats simple normalisation for removing the margin. We support proportional, power and Shin; power was marginally best on our data. |
| Kaunitz, Zhong & Kreiner (2017), [arXiv 1710.02824](https://www.technologyreview.com/2017/10/19/67760/the-secret-betting-strategy-that-beats-online-bookmakers/) | Bet when the best price beats the market-average consensus. Profitable on paper and with real money until bookmakers limited the accounts. Our "consensus" strategy. |
| Winkelmann, Deutscher & Otting (2021), [*Applied Economics*](https://ideas.repec.org/a/taf/applec/v53y2021i26p3054-3064.html); Fischer & Haucap (2021), [DICE](https://www.graduiertenzentrum-medizin.hhu.de/fileadmin/redaktion/Fakultaeten/Wirtschaftswissenschaftliche_Fakultaet/DICE/Bilder/Paper.pdf) | Bookmakers mispriced the vanished home advantage after the COVID break; adaptation was slow. Motivates the self-adjusting home advantage. |
| Winkelmann et al. (2023), *J. Sports Economics*; Angelini & De Angelis (2019), *IJF* | Inefficiencies in the big-five leagues show up in single seasons but are not persistent; efficiency differs across leagues ([summary](https://pub.uni-bielefeld.de/record/2983980)). |
| [Southampton working paper](https://eprints.soton.ac.uk/373110/1/football_20efficiency_20full_20060115.pdf), 52,865 matches, 22 leagues | Efficiency differences between countries are explained by league competitiveness. |

---

## 4. Data sources (status October 2026)

| Source | What it has | Cost / access | Use here |
|---|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk) | Results, shots, corners, cards and odds (Bet365, Pinnacle, market max/avg, closing prices from 2019/20; Pinnacle closing from 2012/13) for 22 European divisions down to the English National League, plus 16 extra leagues. Fixtures with current odds twice a week. | Free CSVs | Primary source (`footy fetch`) |
| [GitHub mirror (xgabora)](https://github.com/xgabora/Club-Football-Match-Data-2000-2025) | Same results and stats, Bet365 + max odds, ClubElo snapshots, 2000 to Sep 2026. No Pinnacle or closing odds; handicap lines rounded. | Free | Fallback (used for this build, since the sandbox could not reach football-data.co.uk) |
| [Understat](https://understat.com) | Shot-level xG for EPL, La Liga, Bundesliga, Serie A, Ligue 1, RPL since 2014. | Free (scrape) | Next step: real xG features for top flights |
| FBref | Lost its Opta advanced stats (xG etc.) on [20 January 2026](https://www.theixsports.com/the-ix-soccer/fbrefs-loss-advanced-stats-womens-soccer-data-accessibility/) after Stats Perform ended the contract. Basic stats remain. | Free | Not used |
| FotMob, Sofascore, WhoScored, Opta Analyst | Opta-based xG and lineups for many leagues ([alternatives after FBref](https://www.valoraanalitik.com/comunidad-futbolera-pierde-plataformas-conocidas-estadisticas-datos-gratis/)). | Free to view; scraping is against their terms | Manual checks |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Full event data for selected competitions. | Free | Research / building an xG model |
| ClubElo | Elo for European clubs since 1940s. | Free | Reference only (snapshots in the mirror) |
| Transfermarkt | Injury lists, suspensions, market values for nearly every professional league, including lower divisions. | Free to view | Team news, squad value as a strength prior |
| [API-Football](https://www.api-football.com) | Fixtures, lineups, injuries, odds for 1,000+ leagues. Free plan around 100 requests/day; paid from about $19/month ([source](https://getbruin.com/docs/ingestr/soccer-sources/api-football.html)). | Freemium | Best value for automated injuries + lineups |
| football-data.org | Fixtures/results API, top leagues on the free tier. | Freemium | Alternative fixture feed |
| [The Odds API](https://the-odds-api.com) and similar | Live odds from many books. | Freemium | Live odds for forward testing |
| Betfair Exchange | Historical data and live API; exchange prices are a sharp reference with commission instead of margin. | Account needed | Best venue for forward testing CLV |
| Pinnacle | Public API [closed on 23 July 2025](https://github.com/pinnacleapi/pinnacleapi-documentation); data now only via partnerships or third-party resellers ([overview](https://odds-api.io/blog/pinnacle-api-shutdown-alternatives)). | Restricted | Closing odds still in football-data.co.uk CSVs |
| Sportmonks, Stats Perform/Opta, Wyscout, StatsBomb | Paid xG, event and player data. | Paid | Only if the edge proves real |

---

## 5. Team news and injury sources

The rating models only see results and shots. Team news is the main thing they
miss, and it moves prices fast. Official lineups are published **75 minutes
before kick-off in the Premier League and 60 minutes before in the EFL**
([EFL regulation 34.5, as summarised here](https://ontourfootball.com/news/confirmed-lineups-when-team-news-actually-drops-and-how-to-get-it-in-time)).
Pre-match press conferences on Thursday/Friday are where most injury news first
appears ([Fantasy Football Scout](https://fantasyfootballscout.co.uk/team-news/)).

### X (Twitter) accounts

Handles change and accounts get impersonated, so check each before relying on
it. A news item is reliable when a national reporter and the club's beat writer
agree.

| Account | Covers | Why |
|---|---|---|
| @premierinjuries (Premier Injuries, Ben Dinnery) | Premier League injuries | Independent injury tracking; the Premier League itself publishes no injury data ([context](https://pocklingtonpost.co.uk/sport/football/more-than-59-years-lost-to-premier-league-alone-in-injuries-this-season-what-can-be-done-4613159)) |
| @David_Ornstein (The Athletic) | Premier League | Most trusted breaking-news reporter in England |
| @FabrizioRomano | Transfers, Europe | Squad changes, new signings |
| @SkySportsNews, @BBCSport | UK football | Press-conference team news |
| @FFScout (Fantasy Football Scout) | Premier League | Collates press-conference injury news and predicted XIs |
| @OfficialFPL | Premier League | Injury flags in the fantasy game update fast |
| @DiMarzio (Gianluca Di Marzio) | Serie A / Serie B | Italian team news |
| @Plettigoal (Florian Plettenberg) | Bundesliga / 2. Bundesliga | German team news |
| @MatteMoretto (Matteo Moretto) | La Liga / Serie A | Spanish and Italian news |
| Official club accounts | Every league | Confirmed XIs at T-75 (PL) / T-60 (EFL) |
| BBC local radio and local-paper club reporters | EFL, National League, Scottish leagues | In lower leagues this is often the only source; build an X List per league |

### Websites

* **Physioroom** and **Premier Injuries**: injury tables (Premier League; Physioroom also covers the EFL).
* **Transfermarkt**: injuries and suspensions for almost every pro league, including tiers 3-5.
* **Fantasy Football Scout team news**, **Rotowire soccer lineups**, **Starting11** ([example](https://starting11.com/fixtures/reading-vs-brentford)): predicted and confirmed XIs.
* **FotMob / Sofascore apps**: confirmed lineups the moment they are released, for lower leagues too.

### Turning news into a number

Enter it in `config/adjustments.yaml` (or the dashboard sliders) as a change in
expected goals. Rough sizes:

| News | Adjustment |
|---|---|
| First-choice striker out | attack -5% to -12% |
| First-choice goalkeeper out | defence +4% to +8% (concede more) |
| Two or more first-choice centre-backs out | defence +6% to +12% |
| Heavy rotation before a cup or European tie | attack -8%, defence +8% |
| New manager | usually already priced in; leave alone |

The edge from news is in **timing**: acting between the news and the price
move. In top flights that window is minutes. In tiers 3-5 it is longer because
fewer people are watching, which is one more reason lower leagues are where to
look.

---

## 6. Why lower divisions, and the risks

For:

* Bookmakers put less work into pricing them; our models get closer to the
  market there (section 1.2).
* Team news comes from a handful of local sources, so it travels slowly.
* Our only persistent lead (O/U 2.5) is in tiers 2-5.

Against:

* Higher margins and worse best prices (2.3% vs 0.9% best-price margin).
* **Low stake limits.** Bookmakers cap lower-league bets at small amounts and
  restrict winning accounts quickly. Exchange liquidity is thin.
* More data errors and stale prices (which inflate backtests).
* Higher integrity risk (match fixing is more common in lower divisions); avoid
  unusual late line moves you cannot explain.
* Less data: fewer shots columns before ~2017 for second tiers outside England.

---

## 7. Recommended next steps

1. **Forward-test the lower-league O/U lead.** Run `footy all` twice a week
   (Tuesday and Friday, when football-data.co.uk updates fixtures and odds),
   log every flagged bet with the price available at that time, and record the
   closing price. Judge it on **CLV over 300-500 bets** before staking money.
2. **Download football-data.co.uk directly** (any machine with internet):
   unlocks Pinnacle and closing odds, so CLV is measured on every backtest bet.
3. **Add real xG** (Understat) for the top flights, and test whether a
   shots-based model beats pseudo-xG in lower leagues once more shot data exists.
4. **Automate team news**: API-Football injuries and lineups (free tier is
   enough for a few leagues) feeding `adjustments.yaml`.
5. **More totals lines**: O/U 1.5 and 3.5 and Asian totals, where margins are
   lower and the same goals model applies.
6. **Stake sizing**: once CLV is positive, use fractional Kelly (0.1-0.25) with
   a per-bet cap; the dashboard's backtest lab shows the bankroll path.

Gambling carries real financial risk; nothing here is a guarantee of future
returns.
