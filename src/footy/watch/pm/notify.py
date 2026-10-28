"""Alert text for prediction-market positions."""

from __future__ import annotations

import json

from . import Signal


def format_signal(s: Signal, row: dict) -> tuple[str, str]:
    legs = json.loads(row["legs"]) if isinstance(row.get("legs"), str) else row.get("legs", [])
    if s.strategy == "consistency":
        title = f"Polymarket arbitrage {100 * s.edge:+.1f}%: {s.title}"
        lines = "\n".join(f"{lg['side']} {lg['label']} @ {100 * lg['price']:.1f}¢ × {lg['shares']:.0f}"
                          for lg in legs[:8]) + (f"\n+{len(legs) - 8} more legs" if len(legs) > 8 else "")
        body = (f"{s.detail}\n{lines}\nCost ${row['cost']:.2f}, pays at least ${row['payout_min']:.2f}. "
                f"Buy every leg, the least liquid first.")
    else:
        lg = legs[0] if legs else s.legs[0]
        title = f"Weather {100 * s.edge:+.0f}%: {lg['side']} {lg['label']} @ {100 * lg['price']:.0f}¢"
        body = f"{s.title}\n{s.detail}\nEdge {100 * s.edge:+.1f}% after the taker fee."
    if s.link:
        body += f"\n{s.link}"
    return title, body
