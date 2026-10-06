"""Iceberg-like absorption detector that uses the full order book depth.

A deal executes against a passive order at some price level. We replay quotes and
deals together, so for every deal we know how much volume the book displayed at that
level just before the trade and how deep the level was (0 = best price).

Three signals are combined (the data has no order ids, so this is level-granular and
cannot tie an episode to a single participant):
  * episode: consecutive deals at one level trade far more than the level ever showed;
  * exceed:  a single deal is larger than the volume displayed at its level, i.e. its
             hidden part executed inside one print (the strongest direct sign);
  * refill:  the level gets refilled within milliseconds after a deal hit it.

Usage: python icebergs.py DATA_DIR INSTRUMENT DAY [DAY ...]
"""
import argparse
import bisect
import os

import numpy as np
import pandas as pd

import qscalp as q

GAP_MS = 30_000        # a pause longer than this ends an episode at a level
LAG_MS = 50            # a quote frame may arrive this much *before* the deal it reflects
AHEAD_MS = 400         # the quotes trail the deals (median ~140 ms, p90 ~330 ms): look this far ahead
REFILL_MAX_MS = 30_000
FAST_MS = 100          # a refill this soon after the level dropped looks mechanical
MIN_EXCESS = 100       # smallest hidden part (contracts) that counts as an exceed
EXCEED_RATIO = 1.5     # deal volume / displayed volume that counts as an exceed


def replay(deals_path, quotes_path):
    """Deals annotated with the displayed volume at their level; refills in ``.attrs``."""
    step = q.read_header(deals_path).price_step
    deals = q.read_deals(deals_path)
    deals = deals[deals["side"].isin([1, 2])].reset_index(drop=True)
    t = q.deals_ms(deals)
    px = (deals["price"] / step).round().astype(int).to_numpy()
    side = deals["side"].to_numpy()
    vol = deals["volume"].to_numpy()
    n = len(deals)
    shown, shown_eff, shown_cons, rank = [0] * n, [0] * n, [0] * n, [0] * n
    win = [0] * n  # largest volume the level showed in the quotes within AHEAD_MS after the deal
    book, upd_t, prev, used = {}, {}, {}, {}
    last_deal, await_dec, dec_t, pending = {}, {}, {}, {}
    refills, lags = [], []

    def take(i, until):
        # settle every deal strictly before ``until`` against the current book
        while i < n and t[i] < until:
            p = px[i]
            v = book.get(p, 0)
            if side[i] == 1:  # aggressor buys -> passive ask (positive volume)
                a = v if v > 0 else 0
                rank[i] = sum(1 for k, x in book.items() if x > 0 and k < p)
            else:  # aggressor sells -> passive bid (negative volume)
                a = -v if v < 0 else 0
                rank[i] = sum(1 for k, x in book.items() if x < 0 and k > p)
            consumed = used.get(p, 0)
            recent = t[i] - upd_t.get(p, -10**18) <= LAG_MS
            cons = max(a, prev.get(p, 0) if recent else 0)
            shown[i] = a                          # raw book volume
            shown_eff[i] = max(a - consumed, 0)   # minus earlier deals not yet reflected in the book
            shown_cons[i] = max(cons - consumed, 0)  # same, also covering a quote that arrived early
            used[p] = consumed + int(vol[i])
            last_deal[p] = t[i]
            await_dec[p] = t[i]
            pending.setdefault(p, []).append(i)
            i += 1
        return i

    i = 0
    for qts, changes in q.iter_quotes(quotes_path):
        i = take(i, qts)
        for p, v in changes:
            old = book.get(p, 0)
            ld = last_deal.get(p)
            pl = pending.get(p)
            if pl:
                pending[p] = [j for j in pl if qts <= t[j] + AHEAD_MS]
                for j in pending[p]:
                    win[j] = max(win[j], abs(v))
            if ld is not None and v != 0 and abs(v) > abs(old) and (old == 0 or (v > 0) == (old > 0)):
                dt = dec_t.pop(p, None)
                # latency counts from the frame in which the level dropped, not from the deal,
                # so the feed lag between deals and quotes cancels out
                if dt is not None and qts - dt <= REFILL_MAX_MS:
                    refills.append((qts, "ask" if v > 0 else "bid", p, qts - dt, abs(v) - abs(old)))
                last_deal.pop(p)
            elif abs(v) < abs(old):
                d = await_dec.pop(p, None)
                if d is not None and qts - d <= 5000:
                    lags.append(qts - d)
                if ld is not None:
                    dec_t[p] = qts
            prev[p] = abs(old)
            upd_t[p] = qts
            used[p] = 0
            if v == 0:
                book.pop(p, None)
            else:
                book[p] = v
    take(i, float("inf"))

    out = deals[["ts", "side", "price", "volume"]].copy()
    out["t"] = t
    out["px"] = px
    out["passive"] = ["ask" if s == 1 else "bid" for s in side]
    out["shown"] = shown
    out["shown_eff"] = shown_eff
    out["shown_cons"] = shown_cons
    out["rank"] = rank
    out["shown_win"] = np.maximum(out["shown_cons"].to_numpy(), np.array(win))
    out["displayed"] = out["shown_win"] > 0
    out["excess"] = out["volume"] - out["shown_win"]
    out["exceed"] = (out["displayed"] & (out["excess"] >= MIN_EXCESS)
                     & (out["volume"] >= EXCEED_RATIO * out["shown_win"]))
    out.attrs["refills"] = pd.DataFrame(refills, columns=["t", "passive", "px", "latency", "added"])
    out.attrs["lags"] = np.array(lags)
    return out


def episodes(trades, gap_ms=GAP_MS):
    """Group deals at the same level and side into episodes (pause > gap_ms splits them)."""
    open_eps, done = {}, []
    cols = zip(trades["t"], trades["passive"], trades["px"], trades["price"], trades["volume"],
               trades["shown"], trades["rank"], trades["exceed"], trades["excess"])
    for t, passive, px, price, vol, shown, rank, exceed, excess in cols:
        key = (passive, px)
        e = open_eps.get(key)
        if e is None or t - e["t1_ms"] > gap_ms:
            if e is not None:
                done.append(e)
            e = {"passive": passive, "px": px, "price": price, "t0_ms": t, "t1_ms": t, "n": 0,
                 "traded": 0, "shown_max": 0, "reloads": 0, "rank": rank, "n_exceed": 0,
                 "exceed_vol": 0, "left": None}
            open_eps[key] = e
        if e["left"] is not None and shown > e["left"]:
            e["reloads"] += 1  # displayed size grew back since the previous trade
        e["left"] = max(shown - vol, 0)
        e["t1_ms"] = t
        e["n"] += 1
        e["traded"] += vol
        e["shown_max"] = max(e["shown_max"], shown)
        if exceed:
            e["n_exceed"] += 1
            e["exceed_vol"] += excess
    done.extend(open_eps.values())
    df = pd.DataFrame(done).drop(columns="left")
    df["ratio"] = df["traded"] / df["shown_max"].clip(lower=1)
    df["t0"] = q._EPOCH + pd.to_timedelta(df["t0_ms"], unit="ms")
    df["t1"] = q._EPOCH + pd.to_timedelta(df["t1_ms"], unit="ms")
    return df


def add_refill_share(eps, refills, fast_ms=FAST_MS, gap_ms=GAP_MS):
    """Per episode: number of refills of its level and the share that came within fast_ms."""
    idx = {}
    for (passive, px), g in refills.groupby(["passive", "px"]):
        g = g.sort_values("t")
        idx[(passive, px)] = (g["t"].to_numpy(), g["latency"].to_numpy())
    n_ref, fast = [], []
    for passive, px, t0, t1 in zip(eps["passive"], eps["px"], eps["t0_ms"], eps["t1_ms"]):
        tt, ll = idx.get((passive, px), (np.array([]), np.array([])))
        lo, hi = np.searchsorted(tt, t0), np.searchsorted(tt, t1 + gap_ms, side="right")
        lat = ll[lo:hi]
        n_ref.append(len(lat))
        fast.append(float((lat <= fast_ms).mean()) if len(lat) else np.nan)
    eps = eps.copy()
    eps["refills"] = n_ref
    eps["fast_share"] = fast
    return eps


def flag(eps, min_vol=300, ratio=4, min_n=15, min_reloads=5):
    return eps[(eps["traded"] >= min_vol) & (eps["ratio"] >= ratio)
               & (eps["n"] >= min_n) & (eps["reloads"] >= min_reloads)]


def _bucket(rank):
    return "best" if rank == 0 else ("2nd-3rd" if rank <= 2 else "deeper")


def _net(df, col):
    bid = df.loc[df["passive"] == "bid", col].sum()
    ask = df.loc[df["passive"] == "ask", col].sum()
    return int(bid), int(ask), int(bid - ask)


def report(day, trades, eps):
    refills = trades.attrs["refills"]
    lags = trades.attrs["lags"]
    print(f"== {day}: deals {len(trades)}, volume {trades['volume'].sum()}")

    # --- signal 2: a single deal bigger than what the level displayed -------------------
    und = trades[~trades["displayed"]]
    ex = trades[trades["exceed"]].copy()
    print(f"-- exceed: deals bigger than the displayed volume (>= {MIN_EXCESS} contracts and "
          f">= {EXCEED_RATIO}x): {len(ex)} deals, hidden part executed {int(ex['excess'].sum())}")
    print(f"   (deals at a level not shown in the book, left out: {len(und)}, volume {int(und['volume'].sum())})")
    b, a, n = _net(ex, "excess")
    print(f"   hidden executed on bids {b} | on asks {a} | net(bid-ask) {n}")
    ex["level"] = ex["rank"].map(_bucket)
    tab = ex.pivot_table(index="level", columns="passive", values="excess", aggfunc="sum",
                         fill_value=0).reindex(["best", "2nd-3rd", "deeper"]).fillna(0).astype(int)
    for c in ("bid", "ask"):
        if c not in tab:
            tab[c] = 0
    tab["net"] = tab["bid"] - tab["ask"]
    print(tab[["bid", "ask", "net"]].to_string())
    top = ex.sort_values("excess", ascending=False).head(5)
    for _, r in top.iterrows():
        print(f"   {r['ts']:%H:%M:%S.%f}"[:-3] + f"  {r['passive']} {r['price']:.3f}  deal {r['volume']} vs shown {r['shown_win']}"
              f"  -> hidden {r['excess']}")

    # --- signal 3: refill speed ---------------------------------------------------------
    if len(lags):
        print(f"-- lag from a deal to the level's decrease in the quotes: median {np.median(lags):.0f} ms, "
              f"p90 {np.percentile(lags, 90):.0f} ms")
    base_fast = float((refills["latency"] <= FAST_MS).mean()) if len(refills) else float("nan")
    print(f"-- refills after a deal: {len(refills)}, share within {FAST_MS} ms of the drop (all levels, baseline): {base_fast:.2f}")

    # --- signal 1 + the two above on the episodes ---------------------------------------
    eps = add_refill_share(eps, refills)
    ice = flag(eps)
    print(f"-- episodes {len(eps)}, iceberg-like by volume/reloads: {len(ice)}")
    tiers = {
        "volume+reloads": ice,
        "  + has exceed deal": ice[ice["n_exceed"] >= 1],
        "  + fast refills (>=50%)": ice[ice["fast_share"] >= 0.5],
        "  + both": ice[(ice["n_exceed"] >= 1) & (ice["fast_share"] >= 0.5)],
    }
    rows = []
    for name, df in tiers.items():
        b, a, n = _net(df, "traded")
        rows.append((name, len(df), b, a, n))
    print(pd.DataFrame(rows, columns=["tier", "episodes", "bid", "ask", "net(bid-ask)"]).to_string(index=False))

    strong = tiers["  + both"].sort_values("traded", ascending=False).head(6)
    if len(strong):
        print("   strongest (all three signals):")
        for _, r in strong.iterrows():
            print(f"   {r['t0']:%H:%M:%S}-{r['t1']:%H:%M:%S} {r['passive']} {r['price']:.3f} deals {r['n']} "
                  f"traded {r['traded']} shown_max {r['shown_max']} exceed {r['n_exceed']} "
                  f"fast {r['fast_share']:.2f}")
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
