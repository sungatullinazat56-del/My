"""Reader for QScalp history files (.qsh, format version 4).

Each file holds one stream (Deals, Quotes or AuxInfo) for one instrument and
one day, gzip-compressed. Prices are stored as integer price steps; the step
is the last field of the instrument code in the header (e.g. ``0.001``).
Timestamps are milliseconds since 0001-01-01; frame timestamps are UTC and are
shifted to Moscow time here, so every returned ``ts`` is Moscow time.
"""
import gzip
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

SIGNATURE = b"QScalp History Data"
GROWING_ESCAPE = 268435455  # 0x0FFFFFFF: the real offset follows as a signed LEB128

STREAM_QUOTES = 0x10
STREAM_DEALS = 0x20
STREAM_AUXINFO = 0x60

_EPOCH = datetime(1, 1, 1)
_MSK_MS = 3 * 3600 * 1000  # frame timestamps are UTC; deal/aux timestamps are already Moscow time


@dataclass
class Header:
    version: int
    application: str
    comment: str
    created_ticks: int
    stream_type: int
    instrument: str
    price_step: float


class _Reader:
    def __init__(self, data, pos=0):
        self.data = data
        self.pos = pos

    def byte(self):
        b = self.data[self.pos]
        self.pos += 1
        return b

    def uleb(self):
        result = shift = 0
        while True:
            b = self.byte()
            result |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                return result

    def leb(self):
        result = shift = 0
        while True:
            b = self.byte()
            result |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                break
        if b & 0x40:
            result |= -(1 << shift)
        return result

    def string(self):
        n = self.uleb()
        s = self.data[self.pos:self.pos + n].decode("utf8")
        self.pos += n
        return s

    def growing(self, last):
        offset = self.uleb()
        if offset == GROWING_ESCAPE:
            offset = self.leb()
        return last + offset

    def relative(self, last):
        return last + self.leb()

    @property
    def eof(self):
        return self.pos >= len(self.data)


def _to_datetime(ms):
    return _EPOCH + timedelta(milliseconds=ms)


def _open(path):
    data = gzip.open(path).read()
    if not data.startswith(SIGNATURE):
        raise ValueError(f"{path}: not a QSH file")
    r = _Reader(data, len(SIGNATURE))
    version = r.byte()
    if version != 4:
        raise ValueError(f"{path}: unsupported QSH version {version}")
    application = r.string()
    comment = r.string()
    created_ticks = struct.unpack_from("<q", data, r.pos)[0]
    r.pos += 8
    if r.byte() != 1:
        raise ValueError(f"{path}: only single-stream files are supported")
    stream_type = r.byte()
    instrument = r.string()
    price_step = float(instrument.rsplit(":", 1)[-1])
    header = Header(version, application, comment, created_ticks, stream_type, instrument, price_step)
    return header, r


def read_header(path):
    return _open(path)[0]


def read_deals(path):
    """Trades: ts, id, side (1 buy / 2 sell / 0 unknown), price, volume, oi, order_id."""
    h, r = _open(path)
    if h.stream_type != STREAM_DEALS:
        raise ValueError(f"{path}: not a Deals stream")
    frame = h.created_ticks // 10000
    ts = deal_id = order_id = price = volume = oi = 0
    rows = []
    while not r.eof:
        frame = r.growing(frame)
        flags = r.byte()
        if flags & 4:
            ts = r.growing(ts)
        if flags & 8:
            deal_id = r.growing(deal_id)
        if flags & 16:
            order_id = r.relative(order_id)
        if flags & 32:
            price = r.relative(price)
        if flags & 64:
            volume = r.leb()
        if flags & 128:
            oi = r.relative(oi)
        rows.append((ts, deal_id, flags & 3, price, volume, oi, order_id))
    df = pd.DataFrame(rows, columns=["ts", "id", "side", "price", "volume", "oi", "order_id"])
    df["ts"] = _EPOCH + pd.to_timedelta(df["ts"], unit="ms")
    df["price"] = (df["price"] * h.price_step).round(10)
    return df


def read_auxinfo(path):
    """Auxiliary info: ts, price, ask_total, bid_total, oi, hi_limit, low_limit."""
    h, r = _open(path)
    if h.stream_type != STREAM_AUXINFO:
        raise ValueError(f"{path}: not an AuxInfo stream")
    frame = h.created_ticks // 10000
    ts = ask = bid = oi = price = hi = low = 0
    rows = []
    while not r.eof:
        frame = r.growing(frame)
        flags = r.byte()
        if flags & 1:
            ts = r.growing(ts)
        if flags & 2:
            ask = r.relative(ask)
        if flags & 4:
            bid = r.relative(bid)
        if flags & 8:
            oi = r.relative(oi)
        if flags & 16:
            price = r.relative(price)
        if flags & 32:
            hi = r.leb()
            low = r.leb()
            r.pos += 8  # deposit (double)
        if flags & 64:
            r.pos += 8  # rate (double)
        if flags & 128:
            r.string()
        rows.append((ts or frame + _MSK_MS, price, ask, bid, oi, hi, low))
    df = pd.DataFrame(rows, columns=["ts", "price", "ask_total", "bid_total", "oi", "hi_limit", "low_limit"])
    df["ts"] = _EPOCH + pd.to_timedelta(df["ts"], unit="ms")
    for c in ("price", "hi_limit", "low_limit"):
        df[c] = (df[c] * h.price_step).round(10)
    return df


def read_quotes(path, depth=1):
    """Order book: top ``depth`` levels per side after every update frame.

    Returns columns ts, bid_{i}, bid_vol_{i}, ask_{i}, ask_vol_{i} for i in 1..depth.
    In the stream, positive volume is ask and negative volume is bid.
    """
    h, r = _open(path)
    if h.stream_type != STREAM_QUOTES:
        raise ValueError(f"{path}: not a Quotes stream")
    frame = h.created_ticks // 10000
    last_price = 0
    book = {}
    rows = []
    while not r.eof:
        frame = r.growing(frame)
        for _ in range(r.leb()):
            last_price = r.relative(last_price)
            volume = r.leb()
            if volume == 0:
                book.pop(last_price, None)
            else:
                book[last_price] = volume
        asks = sorted(p for p, v in book.items() if v > 0)[:depth]
        bids = sorted((p for p, v in book.items() if v < 0), reverse=True)[:depth]
        row = [frame + _MSK_MS]
        for i in range(depth):
            row += [bids[i] * h.price_step if i < len(bids) else None,
                    -book[bids[i]] if i < len(bids) else None,
                    asks[i] * h.price_step if i < len(asks) else None,
                    book[asks[i]] if i < len(asks) else None]
        rows.append(row)
    cols = ["ts"]
    for i in range(1, depth + 1):
        cols += [f"bid_{i}", f"bid_vol_{i}", f"ask_{i}", f"ask_vol_{i}"]
    df = pd.DataFrame(rows, columns=cols)
    df["ts"] = _EPOCH + pd.to_timedelta(df["ts"], unit="ms")
    return df


def iter_quotes(path):
    """Yield ``(ts_ms, changes)`` for every Quotes frame, with the full set of level changes.

    ``ts_ms`` is Moscow-time milliseconds since 0001-01-01 and ``changes`` is a list of
    ``(price_in_steps, signed_volume)``: positive volume is ask, negative is bid, zero
    removes the level. Price step is in ``read_header(path).price_step``.
    """
    h, r = _open(path)
    if h.stream_type != STREAM_QUOTES:
        raise ValueError(f"{path}: not a Quotes stream")
    frame = h.created_ticks // 10000
    last_price = 0
    while not r.eof:
        frame = r.growing(frame)
        changes = []
        for _ in range(r.leb()):
            last_price = r.relative(last_price)
            changes.append((last_price, r.leb()))
        yield frame + _MSK_MS, changes


def deals_ms(deals):
    """Deal timestamps as Moscow-time milliseconds since 0001-01-01 (same scale as iter_quotes)."""
    return ((deals["ts"] - _EPOCH) // pd.Timedelta(milliseconds=1)).to_numpy()


def candles(deals, rule="1min"):
    """OHLCV candles from a deals frame."""
    g = deals.set_index("ts")
    ohlc = g["price"].resample(rule).ohlc()
    ohlc["volume"] = g["volume"].resample(rule).sum()
    return ohlc.dropna(subset=["open"])
