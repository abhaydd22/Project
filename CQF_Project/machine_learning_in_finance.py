# import necessary libraries.
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings("ignore")
import yfinance as yf

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





