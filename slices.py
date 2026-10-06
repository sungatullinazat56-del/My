"""Iceberg detector based on constant slice size in the deals stream.

An iceberg shows a fixed visible part (the peak) and refills it after every fill, so a
passive iceberg that gets hit produces many deals of exactly the same size at one price.
The book only shows the summed volume of a level, which hides this; the deals do not.

A cluster is a run of deals on one side and price with pauses of at most ``gap_ms``. It is
reported when one deal size repeats at least ``min_same`` times and makes up at least
``min_share`` of the cluster volume. This is level-granular: the data has no order ids, so
two participants with equal lots at one price would look like one iceberg.

Usage: python slices.py DATA_DIR INSTRUMENT DAY [DAY ...]
"""
import argparse
import os

import numpy as np
import pandas as pd

import qscalp as q


def clusters(deals, gap_ms=1000, min_same=15, min_share=0.3, min_slice=2):
    d = deals[deals["side"].isin([1, 2])].copy()
    d["t"] = q.deals_ms(d)
    d["passive"] = np.where(d["side"] == 1, "ask", "bid")  # aggressor buy hits the ask
    d = d.sort_values(["passive", "price", "t"]).reset_index(drop=True)
    new = ((d["passive"] != d["passive"].shift()) | (d["price"] != d["price"].shift())
           | (d["t"].diff() > gap_ms))
    d["c"] = new.cumsum()
    rows = []
    for _, g in d.groupby("c"):
        if len(g) < min_same:
            continue
        vc = g["volume"].value_counts()
        size, n = int(vc.index[0]), int(vc.iloc[0])
        share = n * size / g["volume"].sum()
        if n >= min_same and size >= min_slice and share >= min_share:
            rows.append((g["ts"].iloc[0], g["ts"].iloc[-1], g["passive"].iloc[0], g["price"].iloc[0],
                         len(g), int(g["volume"].sum()), size, n, share))
    return pd.DataFrame(rows, columns=["t0", "t1", "passive", "price", "deals", "traded",
                                       "slice", "n_slice", "share"])


def report(day, deals):
    c = clusters(deals)
    bid = c[c["passive"] == "bid"]
    ask = c[c["passive"] == "ask"]
    print(f"== {day}: deals {len(deals)}, volume {int(deals['volume'].sum())}, slice clusters {len(c)}")
    print(f"   passive bid (hidden buying)  {int(bid['traded'].sum()):>9}   in slices {int((bid['slice'] * bid['n_slice']).sum()):>9}")
    print(f"   passive ask (hidden selling) {int(ask['traded'].sum()):>9}   in slices {int((ask['slice'] * ask['n_slice']).sum()):>9}")
    print(f"   net (bid - ask) {int(bid['traded'].sum() - ask['traded'].sum())}")
    for r in c.sort_values("traded", ascending=False).head(8).itertuples():
        print(f"   {r.t0:%H:%M:%S.%f}"[:-3] + f"-{r.t1:%H:%M:%S}  {r.passive} {r.price:.3f}  deals {r.deals}  "
              f"traded {r.traded}  slice {r.slice} x{r.n_slice} ({r.share:.0%})")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("instrument")
    ap.add_argument("days", nargs="+")
    a = ap.parse_args()
    for day in a.days:
        path = os.path.join(a.data_dir, f"{a.instrument}.{day}.Deals.qsh")
        report(day, q.read_deals(path))


if __name__ == "__main__":
    main()
