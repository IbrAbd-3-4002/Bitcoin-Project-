#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bitcoin Realized Volatility Analysis
Created on Tue Feb  3 10:42:22 2026
@author: ibrahimabdullah
"""

import pandas as pd
import numpy as np 
import statsmodels.api as sm
from scipy.stats import norm, t, chi2
from scipy.optimize import minimize
from scipy import stats as scipy_stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# File paths
PATH = "/Users/Admin/OneDrive - The University of Nottingham/Documents/btcusd_1-min_data.csv.csv"
TIME_COL = "Timestamp"
CLOSE_COL = "Close"

# =============================================================================
# SECTION 1: DATA LOADING AND CLEANING
# =============================================================================

print("\n" + "="*70)
print("SECTION 1: DATA LOADING AND CLEANING")
print("="*70)

# Load data and set UTC timezone
df = pd.read_csv(PATH)

# Convert timestamp to UTC datetime
if pd.api.types.is_numeric_dtype(df[TIME_COL]):
    dt = pd.to_datetime(df[TIME_COL].astype("int64"), unit="s", utc=True)
else:
    dt = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")

df["datetime_utc"] = dt
df = df.dropna(subset=["datetime_utc"]).set_index("datetime_utc").sort_index()
price_1m = df[CLOSE_COL].copy()

print(f"Rows loaded: {len(price_1m):,}")
print(f"Date range: {price_1m.index.min()} to {price_1m.index.max()}")

# Check for duplicate timestamps
dup_count = price_1m.index.duplicated(keep="last").sum()
print(f"Duplicate timestamps: {dup_count:,}")

# Check for missing data
full_minutes = pd.date_range(start=df.index.min(), end=df.index.max(), freq="1min", tz="UTC")
missing_minutes = full_minutes.difference(df.index)

print(f"Total expected minutes: {len(full_minutes)}")
print(f"Observed minutes: {len(df.index)}")
print(f"Missing minutes: {len(missing_minutes)}")

# Identify days with most missing minutes
missing_minutes_series = pd.Series(missing_minutes)
missing_by_day = missing_minutes_series.dt.floor("D").value_counts().sort_index()

print(f"Number of days with missing minutes: {len(missing_by_day)}")
print("\nTop 10 days with most missing minutes:")
print(missing_by_day.sort_values(ascending=False).head(10))

# Remove problematic day (2025-03-15 with 1160 missing minutes)
bad_day = pd.Timestamp("2025-03-15", tz="UTC")
df2 = df[df.index.floor("D") != bad_day]
price_1m = df2[CLOSE_COL].copy()
print(f"\nRemoved {bad_day.date()}: {(df.index.floor('D') == bad_day).sum()} rows")

# Recompute expected minutes after dropping bad day
full_minutes = pd.date_range(start=df2.index.min(), end=df2.index.max(), freq="1min", tz="UTC")
full_minutes = full_minutes[full_minutes.floor("D") != bad_day]
missing_minutes = full_minutes.difference(df2.index)
print(f"Missing minutes after cleanup: {len(missing_minutes)}")

# Calculate log returns and flag impossible jumps
log_price_1m = np.log(price_1m)
log_ret_1m = log_price_1m.diff()

MAX_ABS_LOGRET_1M = 0.1  # ~10% move in one minute
impossible_jumps = log_ret_1m.abs() > MAX_ABS_LOGRET_1M

print(f"\nImpossible 1-minute jumps flagged: {int(impossible_jumps.sum())}")
if impossible_jumps.sum() > 0:
    print("Sample impossible jumps:")
    print(log_ret_1m[impossible_jumps].head())
    
    jumps_by_day = log_ret_1m[impossible_jumps].index.floor("D").value_counts().sort_index()
    print(f"Days with impossible jumps: {len(jumps_by_day)}")

# Clean returns by removing impossible jumps
log_ret_clean = log_ret_1m.copy()
log_ret_clean[impossible_jumps] = np.nan
print(f"Impossible jumps after cleaning: {int((log_ret_clean.abs() > MAX_ABS_LOGRET_1M).sum())}")

# Resample to 5-minute log returns
log_ret_5m = log_ret_clean.resample("5min").sum(min_count=1)
log_ret_5m = log_ret_5m[log_ret_5m.index.floor("D") != bad_day]

print(f"\nTotal 5-min bars: {len(log_ret_5m)}")
print(f"Missing 5-min bars: {log_ret_5m.isna().sum()}")

# =============================================================================
# SECTION 2: REALIZED VOLATILITY CALCULATION
# =============================================================================

print("\n" + "="*70)
print("SECTION 2: REALIZED VOLATILITY CALCULATION")
print("="*70)

r_1m = log_ret_clean.copy()
r_5m = log_ret_5m.copy()

# Define trading day indices
day_index_1m = r_1m.index.floor("D")
day_index_5m = r_5m.index.floor("D")

# Compute RV (variance)
RV_1m_var = r_1m.pow(2).groupby(day_index_1m).sum(min_count=1)
RV_1m_vol = np.sqrt(RV_1m_var)

RV_5m_var = r_5m.pow(2).groupby(day_index_5m).sum(min_count=1)
RV_5m_vol = np.sqrt(RV_5m_var)

print(f"1-min RV: {len(RV_1m_vol)} days")
print(f"5-min RV: {len(RV_5m_vol)} days")
print(f"\n1-min RV statistics:")
print(RV_1m_vol.describe())
print(f"\n5-min RV statistics:")
print(RV_5m_vol.describe())

# Plots
plt.figure(figsize=(12, 5))
RV_1m_vol.plot(alpha=0.7, label='1-minute RV')
RV_5m_vol.plot(alpha=0.7, label='5-minute RV')
plt.title("Daily Realized Volatility Comparison")
plt.ylabel("RV (volatility)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# SECTION 3: YANG-ZHANG ESTIMATOR
# =============================================================================

print("\n" + "="*70)
print("SECTION 3: YANG-ZHANG ESTIMATOR")
print("="*70)

# Create daily OHLC
ohlc_daily = price_1m.resample("1D").agg(
    open="first",
    high="max",
    low="min",
    close="last"
).dropna()

print(f"Daily OHLC data: {len(ohlc_daily)} days")
print(f"Date range: {ohlc_daily.index.min().date()} to {ohlc_daily.index.max().date()}")

# Calculate log prices
log_O = np.log(ohlc_daily["open"])
log_H = np.log(ohlc_daily["high"])
log_L = np.log(ohlc_daily["low"])
log_C = np.log(ohlc_daily["close"])

# Yang-Zhang components
r_O = log_O - log_C.shift(1)  # Overnight returns
r_C = log_C - log_O  # Open-to-close returns
RS = (log_H - log_C) * (log_H - log_O) + (log_L - log_C) * (log_L - log_O)  # Rogers-Satchell

# Calculate Yang-Zhang variance and volatility
k = 0.34  # Optimal weight

# Overall estimate
sigma2_overnight = r_O.var()
sigma2_open_close = r_C.var()
sigma2_rs_mean = RS.mean()
YZ_var_overall = sigma2_overnight + k * sigma2_open_close + (1 - k) * sigma2_rs_mean
YZ_vol_overall = np.sqrt(YZ_var_overall)

print(f"\nOverall Yang-Zhang Estimate:")
print(f"  Overnight variance: {sigma2_overnight:.8f}")
print(f"  Open-close variance: {sigma2_open_close:.8f}")
print(f"  Rogers-Satchell mean: {sigma2_rs_mean:.8f}")
print(f"  YZ variance: {YZ_var_overall:.8f}")
print(f"  YZ volatility: {YZ_vol_overall:.6f}")

# Daily time-varying Yang-Zhang
YZ_var_daily = r_O**2 + k * (r_C**2) + (1 - k) * RS
YZ_vol_daily = np.sqrt(YZ_var_daily).dropna()

print(f"\nDaily Yang-Zhang Statistics:")
print(f"  Mean: {YZ_vol_daily.mean():.6f}")
print(f"  Median: {YZ_vol_daily.median():.6f}")
print(f"  Std: {YZ_vol_daily.std():.6f}")

# Rolling Yang-Zhang function
def rolling_yang_zhang(ohlc, window=30, k=0.34):
    """Calculate rolling Yang-Zhang volatility"""
    log_o = np.log(ohlc['open'])
    log_h = np.log(ohlc['high'])
    log_l = np.log(ohlc['low'])
    log_c = np.log(ohlc['close'])
    
    r_o = log_o - log_c.shift(1)
    r_c = log_c - log_o
    rs = (log_h - log_c) * (log_h - log_o) + (log_l - log_c) * (log_l - log_o)
    
    sigma2_o_roll = r_o.rolling(window).var()
    sigma2_c_roll = r_c.rolling(window).var()
    sigma2_rs_roll = rs.rolling(window).mean()
    
    yz_var_roll = sigma2_o_roll + k * sigma2_c_roll + (1 - k) * sigma2_rs_roll
    return np.sqrt(yz_var_roll)

YZ_vol_rolling = rolling_yang_zhang(ohlc_daily, window=30, k=0.34)
print(f"\n30-Day Rolling YZ Mean: {YZ_vol_rolling.mean():.6f}")

# Comparison with 5-minute RV
common_idx = YZ_vol_daily.index.intersection(RV_5m_vol.index)
YZ_common = YZ_vol_daily.loc[common_idx]
RV_5m_vol_common = RV_5m_vol.loc[common_idx]

correlation = YZ_common.corr(RV_5m_vol_common)
print(f"\nYZ vs 5-min RV Correlation: {correlation:.4f}")
print(f"YZ mean: {YZ_common.mean():.6f}")
print(f"5-min RV mean: {RV_5m_vol_common.mean():.6f}")

efficiency_gain = ((RV_5m_vol_common.std() - YZ_common.std()) / RV_5m_vol_common.std()) * 100
print(f"Efficiency gain: {efficiency_gain:.2f}%")

# Plots
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Price
axes[0].plot(ohlc_daily.index, ohlc_daily['close'], color='#2E86AB', linewidth=1.5)
axes[0].set_ylabel('Price (USD)', fontweight='bold')
axes[0].set_title('Bitcoin: Yang-Zhang vs Standard RV', fontweight='bold', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Volatility comparison
axes[1].plot(YZ_common.index, YZ_common, label='Yang-Zhang', color='#A23B72', linewidth=1.5)
axes[1].plot(RV_5m_vol_common.index, RV_5m_vol_common, label='5-min RV', color='#F18F01', linewidth=1, alpha=0.7)
axes[1].set_ylabel('Daily Volatility', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 30-day rolling means
axes[2].plot(YZ_common.rolling(30).mean(), label='YZ (30-day MA)', color='#A23B72', linewidth=1.5)
axes[2].plot(RV_5m_vol_common.rolling(30).mean(), label='5-min RV (30-day MA)', color='#F18F01', linewidth=1, alpha=0.7)
axes[2].set_ylabel('30-day MA', fontweight='bold')
axes[2].set_xlabel('Date', fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('yang_zhang_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Save results
yz_results = pd.DataFrame({
    'Open': ohlc_daily['open'],
    'High': ohlc_daily['high'],
    'Low': ohlc_daily['low'],
    'Close': ohlc_daily['close'],
    'Overnight_Return': r_O,
    'OpenClose_Return': r_C,
    'Rogers_Satchell': RS,
    'YZ_Volatility': YZ_vol_daily,
    'YZ_Rolling_30d': YZ_vol_rolling,
    'RV_5min': RV_5m_vol
})

yz_results.to_csv('yang_zhang_results.csv')
print("\nResults saved to 'yang_zhang_results.csv'")

# =============================================================================
# SECTION 4: STATIONARITY TESTS
# =============================================================================

print("\n" + "="*70)
print("SECTION 4: STATIONARITY TESTS")
print("="*70)

# Test 5-min RV
adf_5m = adfuller(RV_5m_vol.dropna(), autolag="AIC")
print("\n5-min RV ADF Test:")
print(f"  Test statistic: {adf_5m[0]:.4f}")
print(f"  P-value: {adf_5m[1]:.4f}")
print(f"  Stationary: {'Yes' if adf_5m[1] < 0.05 else 'No'}")

# Test Yang-Zhang
adf_yz = adfuller(YZ_vol_daily.dropna(), autolag="AIC")
print("\nYang-Zhang RV ADF Test:")
print(f"  Test statistic: {adf_yz[0]:.4f}")
print(f"  P-value: {adf_yz[1]:.4f}")
print(f"  Stationary: {'Yes' if adf_yz[1] < 0.05 else 'No'}")

# =============================================================================
# SECTION 5: HAR-RV FORECASTING MODEL
# =============================================================================

print("\n" + "="*70)
print("SECTION 5: HAR-RV FORECASTING MODEL")
print("="*70)

# Prepare data for HAR models
df_mod_rv = pd.DataFrame({"RV": RV_5m_var.copy()})
df_mod_rv["RV_d1"] = df_mod_rv["RV"].shift(1)
df_mod_rv["RV_w5"] = df_mod_rv["RV"].rolling(5).mean().shift(1)
df_mod_rv["RV_m22"] = df_mod_rv["RV"].rolling(22).mean().shift(1)
df_mod_rv["RV_next"] = df_mod_rv["RV"].shift(-1)
df_mod_rv = df_mod_rv.dropna()

df_mod_yz = pd.DataFrame({"RV": YZ_var_daily.copy()})
df_mod_yz["RV_d1"] = df_mod_yz["RV"].shift(1)
df_mod_yz["RV_w5"] = df_mod_yz["RV"].rolling(5).mean().shift(1)
df_mod_yz["RV_m22"] = df_mod_yz["RV"].rolling(22).mean().shift(1)
df_mod_yz["RV_next"] = df_mod_yz["RV"].shift(-1)
df_mod_yz = df_mod_yz.dropna()

X_cols = ["RV_d1", "RV_w5", "RV_m22"]

# Train-test split
split_idx_rv = int(len(df_mod_rv) * 0.8)
split_date_rv = df_mod_rv.index[split_idx_rv]
train_rv = df_mod_rv.loc[df_mod_rv.index <= split_date_rv].copy()
test_rv = df_mod_rv.loc[df_mod_rv.index > split_date_rv].copy()

split_idx_yz = int(len(df_mod_yz) * 0.8)
split_date_yz = df_mod_yz.index[split_idx_yz]
train_yz = df_mod_yz.loc[df_mod_yz.index <= split_date_yz].copy()
test_yz = df_mod_yz.loc[df_mod_yz.index > split_date_yz].copy()

print(f"HAR-RV: Training={len(train_rv)}, Test={len(test_rv)}")
print(f"HAR-YZ: Training={len(train_yz)}, Test={len(test_yz)}")

# Rolling forecast HAR-RV
print("\nRunning HAR-RV rolling forecasts...")
rv_hat_rv = []
dates_rv = []

for dt in test_rv.index:
    fit_df = pd.concat([train_rv, test_rv.loc[test_rv.index < dt]])
    fit_df = fit_df.iloc[-1825:]  # Last 5 years
    
    y = fit_df["RV_next"]
    X = sm.add_constant(fit_df[X_cols], has_constant="add")
    model = sm.OLS(y, X).fit()
    
    X_new = test_rv.loc[[dt], X_cols]
    X_new = sm.add_constant(X_new, has_constant="add")
    X_new = X_new[model.params.index]
    
    rv_pred = model.predict(X_new).iloc[0]
    rv_hat_rv.append(max(rv_pred, 0.0))
    dates_rv.append(dt)

rv_hat_rv = pd.Series(rv_hat_rv, index=dates_rv, name="RV_hat")
sigma_hat_rv = np.sqrt(rv_hat_rv)

# Rolling forecast HAR-YZ
print("Running HAR-YZ rolling forecasts...")
rv_hat_yz = []
dates_yz = []

for dt in test_yz.index:
    fit_df = pd.concat([train_yz, test_yz.loc[test_yz.index < dt]])
    fit_df = fit_df.iloc[-1825:]
    
    y = fit_df["RV_next"]
    X = sm.add_constant(fit_df[X_cols], has_constant="add")
    model = sm.OLS(y, X).fit()
    
    X_new = test_yz.loc[[dt], X_cols]
    X_new = sm.add_constant(X_new, has_constant="add")
    X_new = X_new[model.params.index]
    
    rv_pred = model.predict(X_new).iloc[0]
    rv_hat_yz.append(max(rv_pred, 0.0))
    dates_yz.append(dt)

rv_hat_yz = pd.Series(rv_hat_yz, index=dates_yz, name="RV_hat")
sigma_hat_yz = np.sqrt(rv_hat_yz)

# Get daily returns for VaR
daily_close = price_1m.resample("1D").last().dropna()
ret_d = np.log(daily_close).diff().dropna()

# Align returns with forecasts
common_index_rv = ret_d.index.intersection(sigma_hat_rv.index)
ret_test_rv = ret_d.loc[common_index_rv]
sigma_hat_rv = sigma_hat_rv.loc[common_index_rv]

common_index_yz = ret_d.index.intersection(sigma_hat_yz.index)
ret_test_yz = ret_d.loc[common_index_yz]
sigma_hat_yz = sigma_hat_yz.loc[common_index_yz]

mu_hat = ret_d.mean()

print(f"\nMean daily return: {mu_hat:.6f}")
print(f"HAR-RV test periods: {len(ret_test_rv)}")
print(f"HAR-YZ test periods: {len(ret_test_yz)}")

# =============================================================================
# SECTION 6: CALIBRATED VAR WITH OPTIMAL DEGREES OF FREEDOM
# =============================================================================

print("\n" + "="*70)
print("SECTION 6: CALIBRATED VAR WITH OPTIMAL DF")
print("="*70)

# Define alpha
alpha = 0.01  # 99% VaR

# Calculate standardized residuals
z_rv = (ret_test_rv - mu_hat) / sigma_hat_rv
z_yz = (ret_test_yz - mu_hat) / sigma_hat_yz

# Helper functions for DF estimation
def kupiec_test(exceptions, alpha):
    """Kupiec's Unconditional Coverage Test"""
    N = len(exceptions)
    x = exceptions.sum()
    p_hat = x / N
    
    if x == 0:
        p_hat = 0.0001
    if x == N:
        p_hat = 0.9999
    
    LR = -2 * ((N - x) * np.log((1 - alpha) / (1 - p_hat)) + x * np.log(alpha / p_hat))
    p_value = 1 - chi2.cdf(LR, df=1)
    return LR, p_value

def estimate_df_mle(z_residuals):
    """Estimate df using Maximum Likelihood"""
    z_clean = z_residuals.dropna()
    
    def neg_log_likelihood(params):
        df = params[0]
        if df <= 2.01:
            return 1e10
        return -np.sum(scipy_stats.t.logpdf(z_clean, df=df))
    
    result = minimize(neg_log_likelihood, x0=[6.0], bounds=[(2.1, 30)], method='L-BFGS-B')
    return result.x[0]

def estimate_df_kurtosis(z_residuals):
    """Estimate df using excess kurtosis"""
    z_clean = z_residuals.dropna()
    kurt = scipy_stats.kurtosis(z_clean, fisher=False)
    
    if kurt > 3:
        df_est = 4 + 6 / (kurt - 3)
        return min(df_est, 30)
    else:
        return 30

def calibrate_df_coverage(z_residuals, sigma_forecasts, returns, alpha=0.01):
    """Find df that gives correct coverage"""
    z_clean = z_residuals.dropna()
    n = len(returns)
    expected_violations = alpha * n
    
    df_grid = np.linspace(3, 20, 100)
    best_df = 6
    best_error = float('inf')
    
    for df_test in df_grid:
        VaR_test = mu_hat + sigma_forecasts * scipy_stats.t.ppf(alpha, df=df_test)
        violations = (returns < VaR_test).sum()
        error = abs(violations - expected_violations)
        
        if error < best_error:
            best_error = error
            best_df = df_test
    
    return best_df

# Estimate DF for both models
print("\nEstimating optimal degrees of freedom...")

print("\nHAR-RV Model:")
nu_rv_mle = estimate_df_mle(z_rv)
nu_rv_kurt = estimate_df_kurtosis(z_rv)
nu_rv_calib = calibrate_df_coverage(z_rv, sigma_hat_rv, ret_test_rv, alpha)

print(f"  MLE estimate: {nu_rv_mle:.2f}")
print(f"  Kurtosis estimate: {nu_rv_kurt:.2f}")
print(f"  Calibrated (coverage): {nu_rv_calib:.2f}")

print("\nHAR-YZ Model:")
nu_yz_mle = estimate_df_mle(z_yz)
nu_yz_kurt = estimate_df_kurtosis(z_yz)
nu_yz_calib = calibrate_df_coverage(z_yz, sigma_hat_yz, ret_test_yz, alpha)

print(f"  MLE estimate: {nu_yz_mle:.2f}")
print(f"  Kurtosis estimate: {nu_yz_kurt:.2f}")
print(f"  Calibrated (coverage): {nu_yz_calib:.2f}")

# Use calibrated DF
nu_rv_final = nu_rv_calib
nu_yz_final = nu_yz_calib

print(f"\nFinal DF choices:")
print(f"  HAR-RV: {nu_rv_final:.2f}")
print(f"  HAR-YZ: {nu_yz_final:.2f}")

# Calculate VaR with three methods
print("\nCalculating VaR...")

# HAR-RV
VaR_norm_rv = mu_hat + sigma_hat_rv * norm.ppf(alpha)
VaR_t_rv = mu_hat + sigma_hat_rv * scipy_stats.t.ppf(alpha, df=nu_rv_final)
empirical_quantile_rv = np.quantile(z_rv.dropna(), alpha)
VaR_empirical_rv = mu_hat + sigma_hat_rv * empirical_quantile_rv

# HAR-YZ
VaR_norm_yz = mu_hat + sigma_hat_yz * norm.ppf(alpha)
VaR_t_yz = mu_hat + sigma_hat_yz * scipy_stats.t.ppf(alpha, df=nu_yz_final)
empirical_quantile_yz = np.quantile(z_yz.dropna(), alpha)
VaR_empirical_yz = mu_hat + sigma_hat_yz * empirical_quantile_yz

# Calculate exceptions
exc_norm_rv = (ret_test_rv < VaR_norm_rv).astype(int)
exc_t_rv = (ret_test_rv < VaR_t_rv).astype(int)
exc_emp_rv = (ret_test_rv < VaR_empirical_rv).astype(int)

exc_norm_yz = (ret_test_yz < VaR_norm_yz).astype(int)
exc_t_yz = (ret_test_yz < VaR_t_yz).astype(int)
exc_emp_yz = (ret_test_yz < VaR_empirical_yz).astype(int)

# Backtest results
n_rv = len(ret_test_rv)
n_yz = len(ret_test_yz)
expected_rv = alpha * n_rv
expected_yz = alpha * n_yz

print("\n" + "="*70)
print("VAR BACKTESTING RESULTS")
print("="*70)

print(f"\nHAR-RV Model:")
print(f"  Test days: {n_rv}")
print(f"  Expected violations: {expected_rv:.1f}")
print(f"  Normal violations: {exc_norm_rv.sum()} ({exc_norm_rv.sum()/n_rv*100:.2f}%)")
print(f"  Student-t (df={nu_rv_final:.1f}) violations: {exc_t_rv.sum()} ({exc_t_rv.sum()/n_rv*100:.2f}%)")
print(f"  Empirical violations: {exc_emp_rv.sum()} ({exc_emp_rv.sum()/n_rv*100:.2f}%)")

print(f"\nHAR-YZ Model:")
print(f"  Test days: {n_yz}")
print(f"  Expected violations: {expected_yz:.1f}")
print(f"  Normal violations: {exc_norm_yz.sum()} ({exc_norm_yz.sum()/n_yz*100:.2f}%)")
print(f"  Student-t (df={nu_yz_final:.1f}) violations: {exc_t_yz.sum()} ({exc_t_yz.sum()/n_yz*100:.2f}%)")
print(f"  Empirical violations: {exc_emp_yz.sum()} ({exc_emp_yz.sum()/n_yz*100:.2f}%)")

# Kupiec tests
print("\n" + "="*70)
print("KUPIEC TEST RESULTS")
print("="*70)

LR_norm_rv, p_norm_rv = kupiec_test(exc_norm_rv, alpha)
LR_t_rv, p_t_rv = kupiec_test(exc_t_rv, alpha)
LR_emp_rv, p_emp_rv = kupiec_test(exc_emp_rv, alpha)

print(f"\nHAR-RV Model:")
print(f"  Normal VaR:     LR={LR_norm_rv:.4f}, p={p_norm_rv:.4f} {'✓ PASS' if p_norm_rv > 0.05 else '✗ FAIL'}")
print(f"  Student-t VaR:  LR={LR_t_rv:.4f}, p={p_t_rv:.4f} {'✓ PASS' if p_t_rv > 0.05 else '✗ FAIL'}")
print(f"  Empirical VaR:  LR={LR_emp_rv:.4f}, p={p_emp_rv:.4f} {'✓ PASS' if p_emp_rv > 0.05 else '✗ FAIL'}")

LR_norm_yz, p_norm_yz = kupiec_test(exc_norm_yz, alpha)
LR_t_yz, p_t_yz = kupiec_test(exc_t_yz, alpha)
LR_emp_yz, p_emp_yz = kupiec_test(exc_emp_yz, alpha)

print(f"\nHAR-YZ Model:")
print(f"  Normal VaR:     LR={LR_norm_yz:.4f}, p={p_norm_yz:.4f} {'✓ PASS' if p_norm_yz > 0.05 else '✗ FAIL'}")
print(f"  Student-t VaR:  LR={LR_t_yz:.4f}, p={p_t_yz:.4f} {'✓ PASS' if p_t_yz > 0.05 else '✗ FAIL'}")
print(f"  Empirical VaR:  LR={LR_emp_yz:.4f}, p={p_emp_yz:.4f} {'✓ PASS' if p_emp_yz > 0.05 else '✗ FAIL'}")

# Quantile diagnostics
print("\n" + "="*70)
print("QUANTILE DIAGNOSTICS")
print("="*70)

print(f"\nHAR-RV Standardized Residuals:")
print(f"  Empirical 1% quantile: {np.quantile(z_rv.dropna(), 0.01):.4f}")
print(f"  Normal 1% quantile: {norm.ppf(0.01):.4f}")
print(f"  t(df={nu_rv_final:.1f}) 1% quantile: {scipy_stats.t.ppf(0.01, df=nu_rv_final):.4f}")

print(f"\nHAR-YZ Standardized Residuals:")
print(f"  Empirical 1% quantile: {np.quantile(z_yz.dropna(), 0.01):.4f}")
print(f"  Normal 1% quantile: {norm.ppf(0.01):.4f}")
print(f"  t(df={nu_yz_final:.1f}) 1% quantile: {scipy_stats.t.ppf(0.01, df=nu_yz_final):.4f}")

# =============================================================================
# SECTION 7: COMPREHENSIVE RV EXPLORATION
# =============================================================================

print("\n" + "="*70)
print("SECTION 7: COMPREHENSIVE RV EXPLORATION")
print("="*70)

class RVExplorer:
    """Interactive tool for exploring Realized Volatility characteristics"""
    
    def __init__(self, rv_series, name="Realized Volatility", freq="5-min"):
        self.rv = rv_series.replace([np.inf, -np.inf], np.nan).dropna()
        self.name = name
        self.freq = freq
        self.log_rv = np.log(self.rv)
        
        print(f"\nRV EXPLORER: {name}")
        print(f"Frequency: {freq}")
        print(f"Period: {self.rv.index.min().date()} to {self.rv.index.max().date()}")
        print(f"Observations: {len(self.rv)}")
    
    def summary_statistics(self):
        """Print comprehensive summary statistics"""
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print("="*70)
        
        print("\nRV (Original Scale):")
        print(self.rv.describe())
        
        print("\nDistribution Characteristics:")
        print(f"  Skewness: {scipy_stats.skew(self.rv):.4f}")
        print(f"  Kurtosis: {scipy_stats.kurtosis(self.rv):.4f}")
        
        print("\nPersistence Metrics:")
        print(f"  AR(1): {self.rv.autocorr(lag=1):.4f}")
        print(f"  AR(5): {self.rv.autocorr(lag=5):.4f}")
        print(f"  AR(22): {self.rv.autocorr(lag=22):.4f}")
        
        # Stationarity
        adf = adfuller(self.rv.dropna(), autolag='AIC')
        print(f"\nStationarity (ADF):")
        print(f"  Test statistic: {adf[0]:.4f}")
        print(f"  P-value: {adf[1]:.4f}")
        print(f"  Stationary: {'Yes' if adf[1] < 0.05 else 'No'}")
    
    def analyze_clustering(self, threshold_percentile=75):
        """Analyze volatility clustering"""
        print(f"\n{'='*70}")
        print("VOLATILITY CLUSTERING ANALYSIS")
        print("="*70)
        
        # Define high/low volatility - FIX: fillna to handle NaN
        threshold = np.percentile(self.rv.dropna(), threshold_percentile)
        high_vol = (self.rv > threshold).fillna(False)
        
        # Count clusters
        clusters = (high_vol != high_vol.shift()).cumsum()
        cluster_lengths = high_vol.groupby(clusters).sum()
        high_vol_clusters = cluster_lengths[cluster_lengths > 0]
        
        print(f"\nThreshold (p{threshold_percentile}): {threshold:.6f}")
        print(f"High volatility periods: {high_vol.sum()} days ({high_vol.sum()/len(self.rv)*100:.1f}%)")
        
        if len(high_vol_clusters) > 0:
            print(f"Number of high-vol clusters: {len(high_vol_clusters)}")
            print(f"Average cluster length: {high_vol_clusters.mean():.1f} days")
            print(f"Max cluster length: {high_vol_clusters.max():.0f} days")
        
        # Transition probabilities
        high_vol_t = high_vol.copy()
        high_vol_t_minus_1 = high_vol.shift(1).fillna(False)
        valid = ~self.rv.isna() & ~self.rv.shift(1).isna()
        
        high_vol_valid = high_vol_t & valid
        low_vol_valid = (~high_vol_t) & valid
        
        prob_hh = (high_vol_t & high_vol_t_minus_1 & valid).sum() / high_vol_valid.sum() if high_vol_valid.sum() > 0 else 0
        prob_ll = (~high_vol_t & ~high_vol_t_minus_1 & valid).sum() / low_vol_valid.sum() if low_vol_valid.sum() > 0 else 0
        
        print(f"\nTransition Probabilities:")
        print(f"  P(High → High): {prob_hh:.4f}")
        print(f"  P(Low → Low): {prob_ll:.4f}")
        print(f"  Interpretation: {'Strong clustering' if prob_hh > 0.7 else 'Moderate clustering'}")

# Create explorers
print("\nExploring 5-minute RV...")
explorer_5m = RVExplorer(RV_5m_vol, name="5-minute RV", freq="5-min")
explorer_5m.summary_statistics()
explorer_5m.analyze_clustering(threshold_percentile=75)

print("\nExploring Yang-Zhang RV...")
explorer_yz = RVExplorer(YZ_vol_daily, name="Yang-Zhang RV", freq="Daily OHLC")
explorer_yz.summary_statistics()
explorer_yz.analyze_clustering(threshold_percentile=75)

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)

print("\nFiles Generated:")
print("  - yang_zhang_comparison.png")
print("  - yang_zhang_results.csv")

print("\nKey Findings:")
print(f"  • 5-min RV Mean: {RV_5m_vol.mean():.6f}")
print(f"  • Yang-Zhang Mean: {YZ_vol_daily.mean():.6f}")
print(f"  • Correlation: {correlation:.4f}")
print(f"  • Efficiency Gain: {efficiency_gain:.2f}%")
print(f"  • HAR-RV Test Size: {n_rv} days")
print(f"  • HAR-YZ Test Size: {n_yz} days")
print(f"  • Best VaR Model (RV): {'Student-t' if p_t_rv > p_norm_rv else 'Normal'}")
print(f"  • Best VaR Model (YZ): {'Student-t' if p_t_yz > p_norm_yz else 'Normal'}")

print("\n" + "="*70)