# Blaque Baux Breach

**Conditional tail-exceedance regime — does a tail event predict the next one, or just re-describe volatility clustering? Real OOS, but a risk flag, not alpha.**

Breach tests a pre-registered claim: define an **exceedance** as a daily return crossing a fixed quantile threshold,
then condition the forward probability of the *next* exceedance on the current state. Thresholds are **frozen on
2016–2024** and every conditional statistic is read **purely out-of-sample on 2025–26** — the same pre-registration
discipline the family practises (`bombs-away` is the sibling fragility monitor). The honest null: does conditioning on
an exceedance *predict* the next, or merely re-describe vol clustering — and even if it predicts, does it forecast
*direction* (tradable) or only *magnitude* (a risk flag)?

> **Not investment advice.** Educational/research software. See [DISCLAIMER](DISCLAIMER.md) and [LICENSE](LICENSE).

```bash
python3 research/breach_1_exceedance.py   # needs Alpaca data keys in the environment
```

## The finding (real out-of-sample — and a risk signal, not a return one)

[`research/breach_1_exceedance.py`](research/breach_1_exceedance.py) — SPY, frozen 5/95 thresholds, 5-day window.

- **Clustering holds OOS.** After an exceedance, next-5-day exceedance probability is **61% vs 28%** baseline (+33pp) on
  2025–26 — the frozen-param signal survives pre-registered out-of-sample. Disciplined and real.
- **Sharper than slow vol.** A fresh exceedance (73%, full-sample) beats a high-vol *state* (46%) and still lifts the
  probability even when trailing vol is *not* high (53%). A tail event today is a crisper near-term fragility flag than
  trailing volatility — not merely "vol re-described."
- **But magnitude, not direction.** Forward |return| is far bigger after an exceedance (**2.84% vs 1.61%**) while mean
  direction is ~flat (down-exc +0.64% / up-exc +0.60% / base +0.34%). Symmetric clustering forecasts *size*, not *sign*.
- **Not tradable alone.** Buy-the-dip after a down-exceedance earns Sharpe **+0.25 vs SPY +1.03** OOS — the only
  directional scrap is the dip bounce `battered`/`backsliders` already own.

## Verdict

**CONDITIONAL (risk flag) / NULL (alpha).** The conditional-exceedance signal is real and OOS-stable — it holds under
frozen pre-registered parameters and is even a *sharper* near-term fragility flag than a slow high-vol state. That's
genuinely useful — as **risk**. Because clustering is symmetric it forecasts the *size* of the next move, not its
direction, so it is not a standalone return edge. Its honest home is position-sizing and the Layer-3 SafetyGate,
alongside `bombs-away`'s fragility monitor, where a magnitude/clustering forecast earns its keep. A `breach_2` wires it
into sizing, not a direction bet. The pre-registration is the point: **a clean example of a signal that is real *and*
not alpha** — exactly the distinction the family exists to draw.

## Status

**Research / monitor.** Honest conditional — a pre-registered, out-of-sample-validated tail-regime detector, explicitly
scoped to risk rather than return. Frozen-threshold exceedance stats, vol-state control, magnitude-vs-direction test,
a traded falsification, Alpaca SIP daily total return. No live capital.
