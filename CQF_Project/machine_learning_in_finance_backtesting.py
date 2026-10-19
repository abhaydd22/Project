"""
SOXX timing overlay — backtest, PART 2 (CQF Level 2 project, backtesting section).

SEPARATE script; does NOT touch soxx_pipeline.py.  Reuses data_close.pkl and the
SAME feature math as the modelling pipeline.

Diffusion (risk) layer is LOCKED:  rv_20 <= trailing-252d 80th percentile.
This part searches the DRIFT (direction) layer for alpha, testing three
genuinely-different families:

    A  momentum L=10            (baseline)
    C  Hurst-gated momentum     mom>0  AND  hurst>0.5   (trend-regime conditioning)
    D  Variance-ratio-gated     mom>0  AND  vr>1        (trend-regime conditioning)
    E  residual momentum        (SOXX vs QQQ)           (relative strength)

All long-only, point-in-time (signal_{t-1} -> position t), cash earns 0.
Alpha test = ex-post CAPM of strategy returns on SOXX returns.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import statsmodels.api as sm

ANN = np.sqrt(252)     # annualise a DAILY STD / volatility
DPY = 252              # annualise MEANS / CAGR / alpha
CLOSE = "data_close.pkl"

# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def load():
    close = pd.read_pickle(CLOSE)
    soxx = close["SOXX"].astype(float)
    qqq = close["QQQ"].astype(float)
    r = soxx.pct_change()                       # SOXX daily simple returns
    rv20 = r.rolling(20).std() * ANN
    log_s = np.log(soxx).diff()                 # SOXX log returns
    log_q = np.log(qqq).diff()                  # QQQ log returns
    return soxx, r, rv20, log_s, log_q

# ----------------------------------------------------------------------
# features (identical math to soxx_pipeline.py)
# ----------------------------------------------------------------------
def hurst_vr(log_s, win=126, k=20):
    def _hurst(x):
        x = x[~np.isnan(x)]
        if len(x) < 40:
            return np.nan
        d = x - x.mean(); c = np.cumsum(d)
        R = c.max() - c.min(); S = x.std(ddof=1)
        return np.log(R / S) / np.log(len(x)) if S else np.nan

    def _vr(x):
        x = x[~np.isnan(x)]
        if len(x) < 2 * k:
            return np.nan
        rho = np.array([np.corrcoef(x[:-j], x[j:])[0, 1] for j in range(1, k)])
        rho = rho[~np.isnan(rho)]
        return 1.0 + 2.0 * np.sum((1.0 - np.arange(1, len(rho) + 1) / k) * rho)

    h = pd.Series(log_s.rolling(win).apply(_hurst, raw=True), index=log_s.index)
    v = pd.Series(log_s.rolling(win).apply(_vr, raw=True), index=log_s.index)
    return h, v

def residual_momentum(log_s, log_q, k):
    """SOXX vs QQQ beta-residual, accumulated over k days (point-in-time)."""
    beta = log_s.rolling(63).cov(log_q) / log_q.rolling(63).var()
    resid = log_s - beta * log_q
    return resid.rolling(k).sum()

def vol_filter(rv20, q=0.80):
    cap = rv20.rolling(252, min_periods=126).quantile(q)
    return rv20 <= cap

# ----------------------------------------------------------------------
# engine
# ----------------------------------------------------------------------
def apply_signal(long_sig, r):
    """Boolean 'long' series (indexed by date) -> position (entered next day) + returns."""
    pos = long_sig.shift(1).fillna(False).astype(float)   # no look-ahead
    return pos * r, pos

def score(r_strat, r_mkt, pos, cost_bps=0.0):
    x = pd.DataFrame({"s": r_strat, "m": r_mkt}).dropna()
    s, m = x["s"], x["m"]
    n = len(s)
    cagr = (1 + s).prod() ** (DPY / n) - 1
    vol = s.std() * ANN
    sharpe = s.mean() / s.std() * ANN if s.std() > 0 else np.nan
    maxdd = ((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min()
    winrate = (s > 0).mean()

    dpos = pos.diff().abs().fillna(pos)
    turnover = dpos.sum() / n * ANN
    s_net = s - dpos.reindex(x.index).fillna(0) * cost_bps / 1e4
    net_sharpe = s_net.mean() / s_net.std() * ANN if s_net.std() > 0 else np.nan
    if cost_bps > 0:
        s_net = s - dpos.reindex(x.index).fillna(0) * cost_bps / 1e4
    net_sharpe = s_net.mean() / s_net.std() * ANN if s_net.std() > 0 else np.nan
    X = sm.add_constant(m.values)
    ols = sm.OLS(s.values, X).fit()
    alpha = ols.params[0] * DPY
    beta = ols.params[1]
    t_alpha = ols.tvalues[0]
    rsq = ols.rsquared

    return dict(CAGR=cagr, AnnVol=vol, Sharpe=sharpe, MaxDD=maxdd, WinRate=winrate,
                Turnover=turnover, NetSharpe=net_sharpe, Beta=beta, Alpha=alpha,
                tAlpha=t_alpha, R2=rsq, Exposure=pos.mean(), n=n)

def main():
    soxx, r, rv20, log_s, log_q = load()
    print(f"data: SOXX {soxx.index[0].date()} .. {soxx.index[-1].date()} ({len(soxx)} obs)")

    # ---- features ----
    hurst, vr = hurst_vr(log_s)
    vol_ok = vol_filter(rv20)
    mom10 = soxx.pct_change(10) > 0
    resid20 = residual_momentum(log_s, log_q, 20)
    resid60 = residual_momentum(log_s, log_q, 60)

    # ---- configs: diffusion locked (rv_20 filter), sweep the DRIFT layer ----
    configs = {
        "A  baseline mom10":            mom10,
        "C  Hurst-gated (mom & H>0.5)": mom10 & (hurst > 0.5),
        "D  VR-gated (mom & vr>1)":     mom10 & (vr > 1.0),
        "E  resid-mom 20 (SOXX-QQQ)":   resid20 > 0,
        "E  resid-mom 60 (SOXX-QQQ)":   resid60 > 0,
    }

    rows = []
    for name, drift in configs.items():
        sig = drift & vol_ok                      # diffusion filter locked in
        r_strat, pos = apply_signal(sig, r)
        sc = score(r_strat, r, pos)
        sc["Config"] = name
        rows.append(sc)

    # buy-and-hold benchmark
    bh_s = r.fillna(0)
    bh = score(bh_s, bh_s, pd.Series(1.0, index=r.index),)
    bh["Config"] = "B&H SOXX"

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    cols = ["Config", "CAGR", "AnnVol", "Sharpe", "MaxDD", "Exposure",
            "Turnover", "Beta", "Alpha", "tAlpha", "R2"]
    print("\n=== drift search (diffusion = rv_20 filter locked) vs buy-and-hold ===")
    print(res[cols].to_string(index=False))
    print(pd.DataFrame([bh])[cols].to_string(index=False))

    res = pd.concat([res, pd.DataFrame([bh])], ignore_index=True)
    res.to_csv("soxx_backtest_drift_search.csv", index=False)

    # OOS stability + net-cost for the best alpha candidate (rank by Sharpe)
    best = res[res.Config != "B&H SOXX"].sort_values("Sharpe", ascending=False).iloc[0]
    name = best.Config[:40]
    print(f"\n=== OOS sub-period: {best.Config} ===")
    sig = configs[best.Config] & vol_ok
    r_strat, pos = apply_signal(sig, r)
    for a, b in [("2015", "2020"), ("2020", "2025")]:
        x = pd.DataFrame({"s": r_strat, "m": r}).loc[a:b].dropna()
        s, m = x["s"], x["m"]
        o = sm.OLS(s.values, sm.add_constant(m.values)).fit()
        dd = ((1+s).cumprod()/(1+s).cumprod().cummax()-1).min()*100
        print(f"  {a}-{b}: Sharpe {s.mean()/s.std()*ANN:.3f} | alpha {o.params[0]*DPY*100:.2f}%/yr (t={o.tvalues[0]:.2f}) | beta {o.params[1]:.2f} | DD {dd:.1f}%")

    print(f"\n=== net-of-cost sweep: {best.Config} ===")
    for cb in (0, 5, 10, 15):
        sc = score(r_strat, r, pos, cost_bps=cb)
        print(f"  {cb:>2} bps: Sharpe {sc['Sharpe']:.3f} -> net {sc['NetSharpe']:.3f} | CAGR {sc['CAGR']*100:.1f}%")

if __name__ == "__main__":
    main()
