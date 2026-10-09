from datetime import date, datetime, timezone

import pytest

from finsearch.live import collect, grade, keys

UTC = timezone.utc


def test_close_time_and_calendar():
    assert collect.close_time(date(2026, 10, 7)) == datetime(2026, 10, 7, 20, 0, tzinfo=UTC)
    assert collect.close_time(date(2026, 11, 27)) == datetime(2026, 11, 27, 18, 0, tzinfo=UTC)   # half day
    for bad in (date(2026, 10, 10), date(2026, 11, 26)):
        try:
            collect.close_time(bad)
            raise AssertionError("expected SystemExit")
        except SystemExit:
            pass


def test_query_template_filled():
    eq = {"asset": "equity", "name": "EQT", "ticker": "EQT"}
    cr = {"asset": "crypto", "name": "XRP", "ticker": "XRP"}
    t = datetime(2026, 10, 7, 17, 30, tzinfo=UTC)
    assert collect.query_for(eq, "EQT (EQT) closing price {{date}}", "October 7, 2026", t) == \
        "EQT (EQT) closing price October 7, 2026"
    assert collect.query_for(cr, "XRP (XRP) price {{date}} {{time}}", "", t) == "XRP (XRP) price October 7, 2026 17:30 UTC"


def test_crypto_classes():
    target = datetime(2026, 10, 7, 17, 30, tzinfo=UTC)
    ts = int(target.timestamp())

    class Key:
        def minutes(self, tickers, start, end):
            return {"ABC": {ts - 3600: (90.0, 91.0, None), ts - 60: (99.0, 100.0, None), ts: (100.0, 101.0, None),
                            ts + 60: (120.0, 120.1, None)}}

    def r(p):
        return {"ticker": "ABC", "size": "large", "offset_s": 60, "t_sent": ts + 75, "extracted": {"price": p}}

    out = grade.grade_crypto([r(100.2), r(120.05), r(90.4), r(50.0), r(None)], {"target": target.isoformat()}, Key())
    assert [o["class"] for o in out] == ["target", "current", "stale", "other", "none"]
    assert out[0]["correct"] and not out[1]["correct"] and out[2]["implied_age_s"] > 3000


def test_file_coin_source_and_buffer(tmp_path):
    (tmp_path / "live-coins.jsonl").write_text(
        '{"ticker": "XRP", "minute": "2026-10-07T17:29:00Z", "low": 1.0, "high": 1.0, "buffer": 0.01}\n'
        '{"ticker": "XRP", "minute": "2026-10-07T17:30:00Z", "low": 1.0, "high": 1.0, "buffer": 0.01}\n')
    src = keys.coin_source(str(tmp_path))
    ts = int(datetime(2026, 10, 7, 17, 30, tzinfo=UTC).timestamp())
    assert sorted(src.minutes(["XRP"], ts - 60, ts + 60)["XRP"]) == [ts - 60, ts]
    rows = [{"ticker": "XRP", "size": None, "offset_s": 5, "t_sent": ts + 5, "extracted": {"price": 1.009}}]
    out = grade.grade_crypto(rows, {"target": "2026-10-07T17:30:00+00:00"}, src)
    assert out[0]["class"] == "target" and out[0]["buffer"] == 0.01


def test_equity_classes():
    rows = [{"ticker": "X", "extracted": {"price": p, "currency": c}}
            for p, c in [(50.009, "USD"), (48.0, "USD"), (45.0, "EUR"), (49.0, "USD"), (None, None)]]
    out = grade.grade_equity(rows, {"trade_date": "2026-10-07"}, keys.MockCloseSource("50,48"))
    assert [o["class"] for o in out] == ["close", "prior_close", "wrong_currency", "other", "none"]


def test_file_close_source(tmp_path):
    (tmp_path / "live-closes.jsonl").write_text(
        '{"date": "2026-10-07", "ticker": "EQT", "close": 52.31, "prior_close": 52.47}\n')
    src = keys.close_source(str(tmp_path))
    assert src.closes(["EQT", "EPD"], date(2026, 10, 7)) == {
        "EQT": {"close": 52.31, "prior_close": 52.47}, "EPD": {"close": None, "prior_close": None}}
    with pytest.raises(keys.KeyNotAvailable):
        src.closes(["EQT"], date(2026, 10, 8))


def test_plugged_close_source():
    src = keys.close_source("7,6", "finsearch.live.keys:MockCloseSource")
    assert src.closes(["A"], date(2026, 10, 7))["A"] == {"close": 7.0, "prior_close": 6.0}
