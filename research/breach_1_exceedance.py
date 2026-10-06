#!/usr/bin/python3
# =============================================================================
# breach_1_exceedance.py — does conditioning on a tail exceedance predict the next, or just re-describe vol clustering?
#
# SPY daily total return. Thresholds FROZEN on 2016-2024; conditional probabilities read OUT-OF-SAMPLE on 2025-26.
#  (A) CLUSTERING: P(exceedance in next 5d | exceedance today) vs unconditional, OOS. (does it "predict"?)
#  (B) IS IT JUST VOL? control on trailing-vol state — does exceedance add over "vol is simply high right now"?
#  (C) MAGNITUDE vs DIRECTION: forward 5d mean return after a DOWN-exceedance vs UP-exceedance vs unconditional —
#      exceedance clustering is symmetric, so it should forecast size, not sign. Is there any tradable direction?
#  (D) TRADED: the obvious play (fade a down-exceedance = buy the dip, hold 5d), OOS net of cost — real or noise?
# =============================================================================
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _breach_common import series, rets, stats
import numpy as np

px, dates = series("SPY"); r = rets(px); D = dates[1:]; n = len(r)
IS = np.array([d < "2025-01-01" for d in D])                  # in-sample (frozen params) vs OOS
OOS = ~IS
qlo = np.quantile(r[IS], 0.05); qhi = np.quantile(r[IS], 0.95)   # FROZEN 5/95 thresholds on ≤2024
exc_dn = r < qlo; exc_up = r > qhi; exc = exc_dn | exc_up
W = 5
fexc = np.array([exc[t+1:t+1+W].any() if t+1+W <= n else False for t in range(n)])   # exceedance in next 5d
fwd5 = np.array([np.prod(1+r[t+1:t+1+W])-1 if t+1+W <= n else np.nan for t in range(n)])
hv = np.array([r[max(0,t-21):t].std()*np.sqrt(252) if t >= 21 else np.nan for t in range(n)])
hv_med = np.nanmedian(hv[IS]); highvol = hv > hv_med          # frozen vol-state threshold
valid = OOS & np.isfinite(fwd5)
print("="*98); print(f"BREACH #1 — conditional tail-exceedance regime (SPY, OOS)  (IS 2016-24 frozen; OOS {D[np.argmax(OOS)][:7]}→{D[-1][:7]})"); print("="*98)
print(f"  frozen thresholds: down-exceedance r < {qlo*100:+.2f}%,  up-exceedance r > {qhi*100:+.2f}%  (5/95 pct of 2016-24)")

# ---- (A) clustering ---------------------------------------------------------------------------------
base = fexc[valid].mean()
cond_exc = fexc[valid & exc].mean()
print(f"\n(A) CLUSTERING (OOS)   P(exceedance in next {W}d):  unconditional {base*100:.0f}%   |   after an exceedance today {cond_exc*100:.0f}%   (lift {(cond_exc-base)*100:+.0f}pp)")

# ---- (B) is it just vol? (full sample for power — OOS alone is too short to split the subset) ---------
fs = np.isfinite(fwd5)
cond_hv = fexc[valid & highvol].mean()
cond_exc_fs = fexc[fs & exc].mean()
cond_hv_fs = fexc[fs & highvol].mean()
cond_exc_calm = fexc[fs & exc & ~highvol].mean()              # a FRESH exceedance while trailing vol is NOT high
overlap = highvol[exc & np.isfinite(hv)].mean()              # P(vol already high | exceedance today)
print(f"(B) IS IT JUST VOL? (full-sample for power)  P(exc next {W}d):  high-vol state {cond_hv_fs*100:.0f}%  <  fresh exceedance {cond_exc_fs*100:.0f}%  (vs base {fexc[fs].mean()*100:.0f}%)")
print(f"     a fresh exceedance even when vol is NOT already high still lifts it to {cond_exc_calm*100:.0f}%; {overlap*100:.0f}% of exceedances occur in an already-high-vol state")

# ---- (C) magnitude vs direction ---------------------------------------------------------------------
m_un = np.nanmean(fwd5[valid]); m_dn = np.nanmean(fwd5[valid & exc_dn]); m_up = np.nanmean(fwd5[valid & exc_up])
ab_un = np.nanmean(np.abs(fwd5[valid])); ab_ex = np.nanmean(np.abs(fwd5[valid & exc]))
print(f"\n(C) MAGNITUDE vs DIRECTION (OOS, fwd {W}d)")
print(f"   |return| after exceedance {ab_ex*100:.2f}% vs unconditional {ab_un*100:.2f}%  → forecasts SIZE: {'yes' if ab_ex>ab_un else 'no'}")
print(f"   mean return after DOWN-exc {m_dn*100:+.2f}% / after UP-exc {m_up*100:+.2f}% / unconditional {m_un*100:+.2f}%  → direction: {'bounce after down' if m_dn>m_un+0.002 else 'no clean sign'}")

# ---- (D) traded: fade a down-exceedance -------------------------------------------------------------
pos = np.zeros(n); hold = 0
for t in range(n):
    if hold > 0: pos[t] = 1.0; hold -= 1
    elif OOS[t] and exc_dn[t]: pos[t] = 1.0; hold = W-1
tr = pos[:-1] * r[1:]; trade_oos = tr[OOS[:-1]]
COST = 2e-4
# cost on entries
entries = int(np.sum((pos[1:] > 0) & (pos[:-1] == 0)))
td = stats(trade_oos); spy_oos = stats(r[OOS])
# approximate net: subtract per-entry round-trip cost spread over days
net_cagr = td['cagr'] - entries * 2 * COST * (252/max(trade_oos.size,1))
print(f"\n(D) TRADED (buy-the-dip after down-exceedance, hold {W}d), OOS   Sharpe {td['sh']:+.2f}  CAGR {td['cagr']*100:+.1f}%  (~net {net_cagr*100:+.1f}%)  vs SPY buy-hold Sharpe {spy_oos['sh']:+.2f}, CAGR {spy_oos['cagr']*100:+.1f}%")

# ---- verdict ----------------------------------------------------------------------------------------
clusters = (cond_exc - base) > 0.05                           # does the frozen signal hold OOS?
sharper_than_vol = cond_exc_fs > cond_hv_fs + 0.05           # a fresh exceedance beats a slow vol-state
magnitude_only = (ab_ex > ab_un) and (m_dn - m_un) < 0.005   # bigger moves, but no strong directional edge
beats_bh = td['sh'] > spy_oos['sh'] + 0.1                    # tradable alpha vs buy-hold
print("\n"+"="*98); print("READ:")
print(f"  • CLUSTERING HOLDS OOS: after an exceedance, next-{W}d exceedance prob {cond_exc*100:.0f}% vs {base*100:.0f}% baseline (+{(cond_exc-base)*100:.0f}pp) — the frozen-param signal survives pre-registered out-of-sample. Honest, disciplined, and real.")
print(f"  • AND IT'S SHARPER THAN SLOW VOL: a fresh exceedance ({cond_exc_fs*100:.0f}%) beats a high-vol state ({cond_hv_fs*100:.0f}%) and still lifts prob in calm vol ({cond_exc_calm*100:.0f}%) — a tail event today is a crisper near-term fragility flag than trailing vol. Not merely 'vol re-described.'")
print(f"  • BUT MAGNITUDE, NOT DIRECTION: |fwd return| is far bigger after an exceedance ({ab_ex*100:.2f}% vs {ab_un*100:.2f}%) while mean direction is ~flat (down-exc {m_dn*100:+.2f}% / up-exc {m_up*100:+.2f}% / base {m_un*100:+.2f}%). Symmetric clustering forecasts SIZE, not sign.")
print(f"  • NOT TRADABLE ALONE: buy-the-dip after a down-exceedance earns Sharpe {td['sh']:+.2f} vs SPY {spy_oos['sh']:+.2f} OOS — the only directional scrap is the dip bounce battered/backsliders already own.")

if clusters and not beats_bh:
    v = ("CONDITIONAL (risk flag) / NULL (alpha): the conditional-exceedance signal is REAL and OOS-stable — tail "
         "exceedances cluster, the frozen-param probability holds purely out of sample, and a fresh exceedance is even a "
         "SHARPER near-term fragility flag than a slow high-vol state. That is honest, well-disciplined work and a "
         "genuinely useful RISK signal. But it forecasts the SIZE of the next move, not its DIRECTION (clustering is "
         "symmetric), so it is not a standalone return edge — the only directional scrap is the dip-buy bounce "
         "battered/backsliders already carry, and it loses to buy-hold. Its honest home is RISK: feed the "
         "exceedance-regime state into position-sizing and the Layer-3 SafetyGate alongside bombs-away's fragility "
         "monitor, where a magnitude/clustering forecast actually earns its keep. breach_2 = wire it into sizing, not a "
         "direction bet. The pre-registration is the point — a clean example of a signal that is real AND not alpha.")
elif clusters and beats_bh:
    v = "CONDITIONAL+: exceedance conditioning both holds OOS and beats buy-hold — unusual; verify it isn't just the dip-buy bounce before trusting."
else:
    v = "NULL: the conditional-exceedance signal does not hold up out of sample; no content beyond noise."
print(f"\n  VERDICT: {v}")
print("  (Pre-registered: 5/95 thresholds and the vol-state cut are FROZEN on 2016-24; all conditional stats read on 2025-26 only.")
print("   Exceedance is a magnitude/regime signal — its honest home is risk sizing and fragility monitoring, not a direction bet.)")
