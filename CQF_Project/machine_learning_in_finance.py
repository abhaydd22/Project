# import necessary libraries.
import os, io, contextlib, warnings
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings("ignore")
import yfinance as yf
from scipy.signal import savgol_filter
from scipy.stats import multivariate_normal
from hmmlearn.hmm import GaussianHMM
from statsmodels.tsa.stattools import adfuller
from sklearn.feature_selection import mutual_info_classif

# Define the symbols for the financial instruments to be analyzed.
SYMBOLS     = ["SOXX", "QQQ"]
START, END  = "2015-01-01", "2026-10-10"
TRADING_DAYS = 252
CHART_DIR   = "D:/Abhay_New/Abhay/Project/CQF_Project/"
os.makedirs(CHART_DIR, exist_ok=True)

# Download the historical data for the specified symbols using yfinance.
raw = yf.download(SYMBOLS, start=START, end=END, auto_adjust=True, progress=False)
close  = raw["Close"]
volume = raw["Volume"]

# missing values treatment
n_missing = close.isna().sum()
close  = close.dropna()
volume = volume.dropna()

# Check for duplicate indices in the close price data.
n_dup = close.index.duplicated().sum()
if n_dup:
    close = close[~close.index.duplicated(keep="first")]

# Check for duplicate indices in the volume data.
common = close.dropna().index.intersection(volume.dropna().index)
close  = close.loc[common]
volume = volume.loc[common]

# Calculate simple and log returns for the close price data.
simple_ret = close.pct_change().dropna()
log_ret    = np.log(close / close.shift(1)).dropna()

# Standardize the log returns to have a mean of 0 and a standard deviation of 1.
z = (log_ret - log_ret.mean()) / log_ret.std()

# Convert log returns to percentage format for easier interpretation.
log_pct = log_ret * 100
def summarise(s):
    return pd.Series({
        "obs":        s.count(),
        "mean (daily)": s.mean(),
        "std (daily)":  s.std(),
        "min":          s.min(),
        "max":          s.max(),
        "skew":         stats.skew(s.dropna()),
        "excess kurtosis": stats.kurtosis(s.dropna()),
        "ann. mean (%)":  s.mean() * TRADING_DAYS,
        "ann. vol (%)":   s.std() * np.sqrt(TRADING_DAYS),
    })
summary = pd.DataFrame({c: summarise(log_pct[c]) for c in log_pct.columns}).round(3)

# Perform normality tests (Jarque-Bera and Shapiro-Wilk) on the log returns for each symbol.
for sym in SYMBOLS:
    s = log_pct[sym].dropna()
    jb_stat, jb_p = stats.jarque_bera(s)
    sh_w, sh_p = stats.shapiro(s[:5000])       # shapiro caps at ~5000 obs

# Print the results of the normality tests for each symbol.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, sym in zip(axes, SYMBOLS):
    s = log_pct[sym].dropna()
    ax.hist(s, bins=80, density=True, alpha=0.6, color="steelblue", label="returns")
    xs = np.linspace(s.min(), s.max(), 500)
    ax.plot(xs, stats.norm.pdf(xs, s.mean(), s.std()), "r-", lw=2, label="normal fit")
    ax.set_title(f"{sym} daily log-return distribution")
    ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{CHART_DIR}/01_distribution.png", dpi=130); plt.close(fig)

# Generate QQ plots to visually assess the normality of the log returns for each symbol.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, sym in zip(axes, SYMBOLS):
    s = log_pct[sym].dropna()
    stats.probplot(s, dist="norm", plot=ax)
    ax.set_title(f"{sym} — QQ plot vs normal")
    ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{CHART_DIR}/02_qqplot.png", dpi=130); plt.close(fig)

# Define a function to compute the autocorrelation of a time series for a specified number of lags.
def autocorr(x, lags=20):
    x = np.asarray(x, dtype=float)        
    x = x - x.mean()
    denom = (x * x).sum()
    return np.array([(x[:-k] * x[k:]).sum() / denom for k in range(1, lags + 1)])

# Compute the first-order autocorrelation of the log returns and squared log returns for each symbol.
for sym in SYMBOLS:
    r = log_ret[sym].dropna()
    ac1_ret  = autocorr(r, 1)[0]
    ac1_sq   = autocorr(r**2, 1)[0]

# Generate bar plots to visualize the autocorrelation of the log returns and squared log returns for each symbol.
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
n = len(log_ret)
ci = 1.96 / np.sqrt(n)
for col, sym in enumerate(SYMBOLS):
    r = log_ret[sym].dropna()
    for row, (series, title) in enumerate([(r, "returns"), (r**2, "squared returns")]):
        ax = axes[row, col]
        ac = autocorr(series, 20)
        ax.bar(range(1, 21), ac, width=0.6, color="steelblue")
        ax.axhline(ci, ls="--", color="r"); ax.axhline(-ci, ls="--", color="r")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{sym} — {title} autocorrelation"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{CHART_DIR}/03_autocorrelation.png", dpi=130); plt.close(fig)

# Calculate the cumulative growth of $1 invested in each symbol over time based on the simple returns.
cum = (1 + simple_ret).cumprod()
fig, ax = plt.subplots(figsize=(10, 5))
cum.plot(ax=ax, lw=1.6)
ax.set_title("Cumulative growth of $1 (SOXX vs QQQ)")
ax.set_ylabel("growth of $1"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{CHART_DIR}/04_cumulative.png", dpi=130); plt.close(fig)

# Calculate the 63-day rolling annualized volatility of the log returns for each symbol.
roll_vol = log_ret.rolling(63).std() * np.sqrt(TRADING_DAYS) * 100
fig, ax = plt.subplots(figsize=(10, 4.5))
roll_vol.plot(ax=ax, lw=1.4)
ax.set_title("63-day rolling annualised volatility (%)")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{CHART_DIR}/05_rolling_vol.png", dpi=130); plt.close(fig)

# Calculate the monthly log returns and compute the correlation between SOXX and QQQ on monthly bases.  
monthly = np.log(close / close.shift(1)).resample("ME").sum()
corr = monthly["SOXX"].corr(monthly["QQQ"])
corr_daily = log_ret["SOXX"].corr(log_ret["QQQ"])

# Save the processed data (close prices, log returns, simple returns, and volume) to pickle files for future use.
close.to_pickle("data_close.pkl")
log_ret.to_pickle("data_logret.pkl")
simple_ret.to_pickle("data_simpleret.pkl")
volume.to_pickle("data_volume.pkl")

# We'll analyze the features we are going to use for our machine learning model. 
# Configuration: 
T_SOXX      = "SOXX"          
T_QQQ       = "QQQ"           
HORIZON     = 20              # forward horizon (trading days)
THRESHOLD   = 0.0             # up-move threshold on forward return (0 = sign)
RNG         = 42
MACRO_START = "2015-01-01"
ANN = np.sqrt(252)

# Fetch Maro data from Yahoo Finance for the specified symbols and date range.
def fetch_macro(ticker, start=MACRO_START, cache_dir="macro_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    f = os.path.join(cache_dir, ticker.replace("^", "").replace("=", "_") + ".pkl")
    if os.path.exists(f):
        return pd.read_pickle(f)
    import yfinance as yf
    s = yf.download(ticker, start=start, progress=False, auto_adjust=True)["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.astype(float).dropna()
    pd.to_pickle(s, f)
    return s

# Feature 1: multi-horizon momentum + volume breakout. (10, 20 and 60 days) Rolling 20 days volatilty.
def f_momentum(close, volume):
    c = close[T_SOXX]; v = volume[T_SOXX].astype(float)
    out = pd.DataFrame(index=c.index)
    for k in (10, 20, 60):
        out[f"mom_{k}"] = c.pct_change(k)
    out["dvol"] = v / v.rolling(20).mean() - 1.0
    return out

# Feature 2: residual (idiosyncratic) momentum — SOXX stripped of rolling QQQ beta 63 days 
def f_residual_momentum(log_ret):
    sx, qq = log_ret[T_SOXX], log_ret[T_QQQ]
    beta = sx.rolling(63).cov(qq) / qq.rolling(63).var()
    resid = sx - beta * qq
    out = pd.DataFrame(index=log_ret.index)
    for k in (20, 60):
        out[f"resid_mom_{k}"] = resid.rolling(k).sum()
    return out

# Feature 3: Savitzky-Golay velocity / acceleration / phase (centred -> causal lag).
def f_sg_derivatives(close, win=21, poly=3):
    lp = np.log(close[T_SOXX]).values
    vel = savgol_filter(lp, win, poly, deriv=1)
    acc = savgol_filter(lp, win, poly, deriv=2)
    lag = win // 2
    vel = np.concatenate([np.full(lag, np.nan), vel[:-lag]])
    acc = np.concatenate([np.full(lag, np.nan), acc[:-lag]])
    phase = np.where(np.isnan(vel), np.nan,
             np.where((vel > 0) & (acc > 0), 0,
             np.where((vel > 0) & (acc <= 0), 1,
             np.where((vel < 0) & (acc < 0), 2, 3))))
    return pd.DataFrame({"sg_vel": vel, "sg_accel": acc, "sg_phase": phase},
                        index=close.index)

# 4 Hurst (R/S) + Variance Ratio.
def f_hurst_vr(log_ret, win=126, k=20):
    r = log_ret[T_SOXX]

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

    return pd.DataFrame({
        "hurst": r.rolling(win).apply(_hurst, raw=True),
        "vr":    r.rolling(win).apply(_vr, raw=True),
    }, index=log_ret.index)


# 5 realized vol (annualized) + change-in-vol.
def f_realized_vol(log_ret):
    r = log_ret[T_SOXX]
    out = pd.DataFrame(index=log_ret.index)
    for w in (10, 20, 60):
        out[f"rv_{w}"] = r.rolling(w).std() * ANN
    out["drv_20"] = out["rv_20"] - out["rv_20"].shift(20)
    return out


# 6 left-tail reversal (tail5).
def f_tail5(log_ret, win=63, q=0.05):
    r = log_ret[T_SOXX]
    pct = r.rolling(win).quantile(q)
    return pd.DataFrame({
        "tail5": pct,
        "tail_shock": np.where(r < pct, r, 0.0),
    }, index=log_ret.index)

# 7 downside beta — beta estimated on QQQ<0 days only.
def f_downside_beta(log_ret, win=252, minp=60):
    sx, qq = log_ret[T_SOXX], log_ret[T_QQQ]
    down = qq < 0
    cov = sx.where(down).rolling(win, min_periods=minp).cov(qq.where(down))
    var = qq.where(down).rolling(win, min_periods=minp).var()
    return pd.DataFrame({"down_beta": cov / var}, index=log_ret.index)

# 8 3-state Gaussian HMM — expanding refit + forward-filter (no smoothing)
def f_hmm_regime(log_ret, n_states=3, warmup=504, refit=252):
    ret = log_ret[T_SOXX].values.reshape(-1, 1)
    obs = np.hstack([ret, np.abs(ret)])
    T = len(ret)

    def _emit(means, covs, o):
        return np.array([multivariate_normal.pdf(o, means[k], covs[k])
                         for k in range(len(means))])

    def _init(start, means, covs, o):
        a = start * _emit(means, covs, o); return a / a.sum()

    def _step(a, trans, means, covs, o):
        a = a @ trans * _emit(means, covs, o); s = a.sum()
        return a / s if s > 0 else a

    bull, state = np.full(T, np.nan), np.full(T, np.nan)
    bull_s, last_refit, alpha = 0, warmup, None
    for t in range(warmup, T):
        if t == warmup or (t - last_refit) >= refit:
            m = GaussianHMM(n_components=n_states, covariance_type="diag",
                            n_iter=100, tol=1e-4, random_state=RNG)
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                m.fit(obs[:t])
            start, trans, means, covs = m.startprob_, m.transmat_, m.means_, m.covars_
            bull_s = int(np.argmax(means[:, 0]))
            alpha = _init(start, means, covs, obs[t])
            last_refit = t
        else:
            alpha = _step(alpha, trans, means, covs, obs[t])
        bull[t] = alpha[bull_s]; state[t] = int(np.argmax(alpha))
    return pd.DataFrame({"hmm_bull": bull, "hmm_state": state}, index=log_ret.index)

# 9 Kalman local-linear-trend slope + random-walk beta (forward filters). 
def f_kalman(close, log_ret):
    def _trend(log_close, q_level=1e-6, q_slope=1e-8):
        F = np.array([[1., 1.], [0., 1.]]); H = np.array([[1., 0.]])
        Q = np.diag([q_level, q_slope])
        r = max(1e-6, float(np.var(np.diff(log_close))) * 0.01)
        x, P = np.zeros(2), np.eye(2) * 1e3
        slopes = np.full(len(log_close), np.nan)
        for i, z in enumerate(log_close):
            x = F @ x; P = F @ P @ F.T + Q
            y = z - (H @ x)[0]; S = (H @ P @ H.T)[0, 0] + r
            K = (P @ H.T)[:, 0] / S
            x = x + K * y; P = (np.eye(2) - np.outer(K, H[0])) @ P
            slopes[i] = x[1]
        return slopes

    def _beta(sx, qq, q=1e-4):
        r = max(1e-6, float(np.var(sx)) * 0.05)
        x, P = 1.0, 1.0
        betas = np.full(len(sx), np.nan)
        for i in range(len(sx)):
            h = qq[i]
            if np.isnan(h):
                betas[i] = x; continue
            P = P + q; S = h * h * P + r; K = P * h / S
            x = x + K * (sx[i] - h * x); P = (1.0 - K * h) * P
            betas[i] = x
        return betas

    out = pd.DataFrame(index=close.index)
    out["kalman_trend"] = _trend(np.log(close[T_SOXX]).values)
    out["kalman_beta"] = pd.Series(_beta(log_ret[T_SOXX].values, log_ret[T_QQQ].values),
                                   index=log_ret.index)
    return out

# 10 SOXX-QQQ rolling OLS spread + z-score (ADF/half-life as diagnostics).
def f_cointegration(close, reg_win=252, z_win=63):
    lx, ly = np.log(close[T_QQQ]), np.log(close[T_SOXX])
    beta = ly.rolling(reg_win).cov(lx) / lx.rolling(reg_win).var()
    alpha = ly.rolling(reg_win).mean() - beta * lx.rolling(reg_win).mean()
    spread = ly - (alpha + beta * lx)
    z = (spread - spread.rolling(z_win).mean()) / spread.rolling(z_win).std()

    # full-sample diagnostics (printed, not features)
    fs = (ly - ly.mean()) - np.polyfit(lx, ly, 1)[0] * (lx - lx.mean())
    adf = adfuller(fs.dropna(), autolag="AIC")
    rl = np.polyfit(fs.shift(1).dropna(), fs.diff().dropna(), 1)[0]
    hl = -np.log(2) / rl if rl < 0 else np.inf
    print(f"   [coint diagnostic] ADF={adf[0]:.3f} p={adf[1]:.4f}  half-life={hl:.1f}d")

    return pd.DataFrame({"spread": spread, "spread_z": z, "coint_beta": beta},
                        index=close.index)

# 11 VIX level + daily change.
def f_vix(close):
    vix = fetch_macro("^VIX").reindex(close.index, method="ffill")
    return pd.DataFrame({"vix": vix, "dvix": vix.diff()}, index=close.index)

# 12 10Y Treasury yield + daily change.
def f_treasury(close):
    y = fetch_macro("^TNX").reindex(close.index, method="ffill")
    return pd.DataFrame({"y10": y, "dy10": y.diff()}, index=close.index)

# 13 USD index level + daily return.
def f_dxy(close):
    dxy = fetch_macro("DX-Y.NYB")
    if len(dxy) == 0:
        dxy = fetch_macro("DX=F")
    dxy = dxy.reindex(close.index, method="ffill")
    return pd.DataFrame({"dxy": dxy, "rdxy": dxy.pct_change()}, index=close.index)

def build_target(close):
    c = close[T_SOXX]
    fwd = c.shift(-HORIZON) / c - 1.0
    y = (fwd > THRESHOLD).astype(float)
    y[fwd.isna()] = np.nan
    return pd.DataFrame({"fwd_ret": fwd, "y": y}, index=close.index)

# Compute all 13 features + target, join on the date index.
def build_matrix(close, log_ret, simple_ret=None, volume=None):
    idx = close.index
    parts = [
        f_momentum(close, volume),
        f_residual_momentum(log_ret),
        f_sg_derivatives(close),
        f_hurst_vr(log_ret),
        f_realized_vol(log_ret),
        f_tail5(log_ret),
        f_downside_beta(log_ret),
        f_hmm_regime(log_ret),
        f_kalman(close, log_ret),
        f_cointegration(close),
        f_vix(close),
        f_treasury(close),
        f_dxy(close),
    ]
    mat = pd.concat(parts, axis=1).reindex(idx)
    tgt = build_target(close)
    mat = pd.concat([mat, tgt], axis=1)
    feat = [c for c in mat.columns if c not in ("y", "fwd_ret")]
    return mat[feat + ["fwd_ret", "y"]], feat

# Mutual-information filter (first funnel stage).
def mi_filter(matrix):
    feat = [c for c in matrix.columns if c not in ("y", "fwd_ret")]
    X = matrix[feat].ffill().bfill()
    y = matrix["y"]
    valid = y.notna() & X.notna().all(axis=1)
    X, y = X[valid].values, y[valid].astype(int).values
    mi = mutual_info_classif(X, y, random_state=RNG)
    rank = pd.DataFrame({"feature": feat, "MI": mi}).sort_values("MI", ascending=False)
    rank["rank"] = range(1, len(rank) + 1)
    return rank, len(y)

def main():
    # try to use DataFrames already in scope; else load the four pickles
    g = globals()
    close   = g.get("close")   or (pd.read_pickle("data_close.pkl")     if os.path.exists("data_close.pkl")     else None)
    log_ret = g.get("log_ret") or (pd.read_pickle("data_logret.pkl")    if os.path.exists("data_logret.pkl")    else None)
    volume  = g.get("volume")  or (pd.read_pickle("data_volume.pkl")    if os.path.exists("data_volume.pkl")    else None)
    if close   is None and os.path.exists("data_close.pkl"):  close   = pd.read_pickle("data_close.pkl")
    if log_ret is None and os.path.exists("data_logret.pkl"): log_ret = pd.read_pickle("data_logret.pkl")
    if volume  is None and os.path.exists("data_volume.pkl"): volume  = pd.read_pickle("data_volume.pkl")
    
    matrix, feat = build_matrix(close, log_ret, None, volume)
    matrix.to_pickle("feature_matrix.pkl")
    matrix.to_csv("feature_matrix.csv")

    yv = matrix["y"].dropna()
    print(f"=== FEATURE MATRIX: {matrix.shape[0]} x {matrix.shape[1]} "
          f"({len(feat)} features) ===")
    print(f"target balance: P(up)={yv.mean():.3f}  up={int(yv.sum())} down={int((1-yv).sum())}")

    rank, n = mi_filter(matrix)
    rank.to_csv("mi_ranking.csv", index=False)
    print(f"\n=== MI FILTER (n={n}) ===")
    print(rank.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return matrix, rank


if __name__ == "__main__":
    main()





