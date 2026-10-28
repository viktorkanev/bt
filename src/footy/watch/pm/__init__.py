"""Prediction-market strategies that don't need a bookmaker feed.

* ``consistency`` - prices on Polymarket that contradict each other: the outcomes of
  one event that don't add up to $1, or a ladder ("above $100k", "above $110k";
  "by March", "by June") where a narrower outcome costs more than a wider one.
  Buying the right combination locks in a profit whatever happens.
* ``weather`` - daily temperature markets priced against weather-model ensembles
  (Open-Meteo) and, on the day itself, the highest temperature already observed at
  the airport the market settles on (METAR).

Both record imaginary positions in their own ledger (``ledger.PMBook``) and settle
them from Polymarket's own resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Signal:
    strategy: str                 # "consistency" | "weather"
    key: str                      # one position per key, never repeated
    event_id: str
    title: str
    slug: str
    end_date: datetime | None
    kind: str                     # "all-yes" | "all-no" | "ladder" | "weather-yes" | "weather-no"
    legs: list[dict]              # market_id, label, token, side ("YES"/"NO"), price, fee, size (shares at price)
    payout_per_set: float | None  # guaranteed payout of one share of every leg (arbitrage); None if not guaranteed
    prob: float | None = None     # model probability that a single-leg position pays out
    edge: float = 0.0             # expected profit per dollar
    apy: float | None = None      # edge per year until resolution
    detail: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def link(self) -> str | None:
        return f"https://polymarket.com/event/{self.slug}" if self.slug else None


def build_scanners(cfg: dict) -> list:
    out = []
    if cfg.get("consistency", {}).get("enabled"):
        from .consistency import ConsistencyScanner
        out.append(ConsistencyScanner(cfg))
    if cfg.get("weather", {}).get("enabled"):
        from .weather import WeatherScanner
        out.append(WeatherScanner(cfg))
    return out
