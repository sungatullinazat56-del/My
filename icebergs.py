"""Iceberg-like absorption detector that uses the full order book depth.

A deal executes against a passive order at some price level. We replay quotes and
deals together, so for every deal we know how much volume the book displayed at that
level just before the trade and how deep the level was (0 = best price). Consecutive
deals at one level form an episode. An episode looks like an iceberg when far more
volume traded at the level than it ever displayed, and the displayed size kept
refilling between trades. This works at level granularity: it cannot tie an episode
to a single participant.

Usage: python icebergs.py DATA_DIR INSTRUMENT DAY [DAY ...]
"""
import argparse
import os

import pandas as pd

import qscalp as q

GAP_MS = 30_000  # a pause longer than this ends an episode at a level


def replay(deals_path, quotes_path):
    """Deals with the displayed volume at their price before the trade and the level rank."""
    step = q.read_header(deals_path).price_step
    deals = q.read_deals(deals_path)
    deals = deals[deals["side"].isin([1, 2])].reset_index(drop=True)
    t = q.deals_ms(deals)
    px = (deals["price"] / step).round().astype(int).to_numpy()
    side = deals["side"].to_numpy()
    n = len(deals)
    shown = [0] * n
    rank = [0] * n
    book = {}
    i = 0

    def take(i, until):
        # settle every deal strictly before ``until`` against the current book
        while i < n and t[i] < until:
            p = px[i]
            v = book.get(p, 0)
            if side[i] == 1:  # aggressor buys -> passive ask (positive volume)
                shown[i] = v if v > 0 else 0
                rank[i] = sum(1 for k, x in book.items() if x > 0 and k < p)
            else:  # aggressor sells -> passive bid (negative volume)
                shown[i] = -v if v < 0 else 0
                rank[i] = sum(1 for k, x in book.items() if x < 0 and k > p)
            i += 1
        return i

    for qts, changes in q.iter_quotes(quotes_path):
        i = take(i, qts)
        for p, v in changes:
            if v == 0:
                book.pop(p, None)
            else:
                book[p] = v
    take(i, float("inf"))
    out = deals[["ts", "side", "price", "volume"]].copy()
    out["t"] = t
    out["passive"] = ["ask" if s == 1 else "bid" for s in side]
    out["shown"] = shown
    out["rank"] = rank
    return out


def episodes(trades, gap_ms=GAP_MS):
    """Group deals at the same level and side into episodes (pause > gap_ms splits them)."""
    open_eps, done = {}, []
    for t, passive, price, vol, shown, rank in zip(
            trades["t"], trades["passive"], trades["price"], trades["volume"],
            trades["shown"], trades["rank"]):
        key = (passive, price)
        e = open_eps.get(key)
        if e is None or t - e["t1"] > gap_ms:
            if e is not None:
                done.append(e)
            e = {"passive": passive, "price": price, "t0": t, "t1": t, "n": 0, "traded": 0,
                 "shown_max": 0, "reloads": 0, "rank": rank, "left": None}
            open_eps[key] = e
        if e["left"] is not None and shown > e["left"]:
            e["reloads"] += 1  # displayed size grew back since the previous trade
        e["left"] = max(shown - vol, 0)
        e["t1"] = t
        e["n"] += 1
        e["traded"] += vol
        e["shown_max"] = max(e["shown_max"], shown)
    done.extend(open_eps.values())
    df = pd.DataFrame(done).drop(columns="left")
    df["ratio"] = df["traded"] / df["shown_max"].clip(lower=1)
    for c in ("t0", "t1"):
        df[c] = q._EPOCH + pd.to_timedelta(df[c], unit="ms")
    return df


def flag(eps, min_vol=300, ratio=4, min_n=15, min_reloads=5):
    return eps[(eps["traded"] >= min_vol) & (eps["ratio"] >= ratio)
               & (eps["n"] >= min_n) & (eps["reloads"] >= min_reloads)]


def _bucket(rank):
    return "best" if rank == 0 else ("2nd-3rd" if rank <= 2 else "deeper")


def report(day, trades, eps):
    ice = flag(eps)
    print(f"== {day}: deals {len(trades)}, volume {trades['volume'].sum()}, episodes {len(eps)}, "
          f"iceberg-like {len(ice)}")
    ice = ice.assign(level=ice["rank"].map(_bucket))
    tab = ice.pivot_table(index="level", columns="passive", values="traded", aggfunc="sum",
                          fill_value=0).reindex(["best", "2nd-3rd", "deeper"]).fillna(0).astype(int)
    for c in ("bid", "ask"):
        if c not in tab:
            tab[c] = 0
    tab["net(bid-ask)"] = tab["bid"] - tab["ask"]
    tab.loc["total"] = tab.sum()
    print(tab[["bid", "ask", "net(bid-ask)"]].to_string())
    hours = ice.assign(hour=ice["t0"].dt.floor("h"))
    h = hours.pivot_table(index="hour", columns="passive", values="traded", aggfunc="sum", fill_value=0)
    h = h.reindex(columns=["bid", "ask"], fill_value=0).astype(int)
    h["net"] = h["bid"] - h["ask"]
    h["cum_net"] = h["net"].cumsum()
    h.index = h.index.strftime("%H:00")
    print(h.to_string())
    top = ice.sort_values("traded", ascending=False).head(6)[
        ["passive", "price", "t0", "t1", "n", "traded", "shown_max", "reloads", "rank"]]
    print(top.to_string(index=False))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("instrument")
    ap.add_argument("days", nargs="+")
    a = ap.parse_args()
    for day in a.days:
        base = os.path.join(a.data_dir, f"{a.instrument}.{day}.")
        trades = replay(base + "Deals.qsh", base + "Quotes.qsh")
        report(day, trades, episodes(trades))


if __name__ == "__main__":
    main()
