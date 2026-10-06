#!/usr/bin/python3
# =============================================================================
# _breach_common.py — shared helpers for Blaque Baux BREACH (conditional tail-exceedance regime, OOS).
# Alpaca SIP daily bars (adjustment=all → total return); env keys. Read-only.
#
# BREACH tests a pre-registered claim: define EXCEEDANCE events as daily/horizon returns crossing a fixed quantile
# threshold, then condition the forward probability of the NEXT exceedance on the current state. Thresholds are
# FROZEN on data through 2024 and the conditional probabilities are read PURELY out-of-sample (2025-26) — the same
# pre-registration discipline the family practices (bombs-away is the sibling fragility monitor). The honest null:
# does conditioning on a tail exceedance actually PREDICT the next exceedance, or does it merely re-describe
# volatility clustering (big moves cluster → P(exceedance | recent exceedance) is trivially high)? And even if the
# probability is elevated, exceedance clustering is SYMMETRIC — it forecasts MAGNITUDE (more big moves), not
# DIRECTION, so it is a risk/regime flag, not by itself a tradable return edge. We separate those two questions.
# =============================================================================
import os, json, urllib.request, math
import numpy as np

H = {"APCA-API-KEY-ID": os.environ["ALPACA_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"]}
START, END = "2016-01-01", "2026-08-01"
_cache = {}

def bars(s):
    if s in _cache: return _cache[s]
    u = (f"https://data.alpaca.markets/v2/stocks/bars?symbols={s}&timeframe=1Day"
         f"&start={START}&end={END}&adjustment=all&feed=sip&limit=10000")
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=40))
        _cache[s] = {b["t"][:10]: b["c"] for b in d.get("bars", {}).get(s, [])}
    except Exception:
        _cache[s] = {}
    return _cache[s]

def series(s):
    b = bars(s); dates = sorted(b)
    return np.array([b[d] for d in dates], float), dates

def rets(px): return px[1:] / px[:-1] - 1

def stats(r):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    if len(r) < 20 or r.std() == 0: return dict(sh=float('nan'), cagr=float('nan'), dd=float('nan'), vol=float('nan'))
    cum = np.cumprod(1 + r)
    return dict(sh=r.mean()/r.std()*math.sqrt(252), cagr=cum[-1]**(252/len(r))-1,
                dd=float((cum/np.maximum.accumulate(cum)-1).min()), vol=r.std()*math.sqrt(252))
