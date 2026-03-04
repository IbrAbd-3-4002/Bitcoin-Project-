import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from scipy.stats import norm, t
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Bitcoin Realized Volatility Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-left: 5px solid #1E88E5;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-left: 5px solid #FF9800;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# VALUE AT RISK (VaR) FUNCTION
# =============================================================================

def compute_var(
    returns: pd.Series,
    sigma: pd.Series,
    alpha: float = 0.01,
    dist: str = "Normal",
    df_t: int = 8,
    use_mu: bool = False
) -> pd.Series:
    """
    1-day parametric VaR for a LONG position (loss is in left tail).
    VaR_t is a negative number (e.g., -0.05 means -5%).
    """
    r = returns.reindex(sigma.index).dropna()
    s = sigma.reindex(r.index).dropna()
    mu = r.rolling(252).mean() if use_mu else 0.0
    mu = mu if isinstance(mu, pd.Series) else pd.Series(mu, index=s.index)

    if dist.lower().startswith("norm"):
        q = norm.ppf(alpha)
    else:
        q = t.ppf(alpha, df=df_t)

    var = mu + s * q
    return var

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

@st.cache_data
def load_and_prepare_data():
    """Load and prepare all RV series"""
    try:
        # Update this path to your actual data file
        PATH = "/Users/Admin/OneDrive - The University of Nottingham/Documents/btcusd_1-min_data.csv.csv"
        TIME_COL = "Timestamp"
        CLOSE_COL = "Close"
        
        df = pd.read_csv(PATH)
        
        # Convert timestamp to UTC datetime
        if pd.api.types.is_numeric_dtype(df[TIME_COL]):
            dt = pd.to_datetime(df[TIME_COL].astype("int64"), unit="s", utc=True)
        else:
            dt = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")
        
        df["datetime_utc"] = dt
        df = df.dropna(subset=["datetime_utc"]).set_index("datetime_utc").sort_index()
        
        # Use 1 min close price only
        price_1m = df[CLOSE_COL].copy()
        
        print("Rows loaded:", f"{len(price_1m):,}")
        print("Date range:", price_1m.index.min(), "to", price_1m.index.max())
        
        # Calculate 1-minute log returns
        log_price_1m = np.log(price_1m)
        log_ret_1m = log_price_1m.diff()
        
        # Remove impossible jumps
        MAX_ABS_LOGRET_1M = 0.1
        impossible_jumps = log_ret_1m.abs() > MAX_ABS_LOGRET_1M
        log_ret_clean = log_ret_1m.copy()
        log_ret_clean[impossible_jumps] = np.nan
        
        # Calculate 5-minute log returns
        log_ret_5m = log_ret_clean.resample("5min").sum(min_count=1)
        
        # Calculate daily RV
        day_index_1m = log_ret_clean.index.floor("D")
        day_index_5m = log_ret_5m.index.floor("D")
        
        RV_1m_var = log_ret_clean.pow(2).groupby(day_index_1m).sum(min_count=1)
        RV_1m_vol = np.sqrt(RV_1m_var)
        
        RV_5m_var = log_ret_5m.pow(2).groupby(day_index_5m).sum(min_count=1)
        RV_5m_vol = np.sqrt(RV_5m_var)
        
        # Calculate Yang-Zhang
        ohlc_daily = price_1m.resample("1D").agg(
            open="first",
            high="max",
            low="min",
            close="last"
        ).dropna()
        
        log_O = np.log(ohlc_daily["open"])
        log_H = np.log(ohlc_daily["high"])
        log_L = np.log(ohlc_daily["low"])
        log_C = np.log(ohlc_daily["close"])
        
        r_O = log_O - log_C.shift(1)
        r_C = log_C - log_O
        RS = (log_H - log_C) * (log_H - log_O) + (log_L - log_C) * (log_L - log_O)
        
        YZ_var_daily = r_O**2 + 0.34 * (r_C**2) + (1 - 0.34) * RS
        YZ_vol_daily = np.sqrt(YZ_var_daily).dropna()
        
        returns_daily = np.log(price_1m.resample("1D").last()).diff().dropna()
        
        return {
            'RV_1m_vol': RV_1m_vol,
            'RV_5m_vol': RV_5m_vol,
            'YZ_vol_daily': YZ_vol_daily,
            'price_1m': price_1m,
            'returns_daily': returns_daily
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Using sample generated data for demonstration...")
        
        # Return dummy data for demonstration with timezone-aware index
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D', tz='UTC')
        RV_1m_vol = pd.Series(np.random.gamma(2, 0.02, len(dates)) + 0.01, index=dates)
        RV_5m_vol = pd.Series(np.random.gamma(2, 0.015, len(dates)) + 0.008, index=dates)
        YZ_vol_daily = pd.Series(np.random.gamma(2, 0.018, len(dates)) + 0.009, index=dates)
        price_1m = pd.Series(np.cumprod(1 + np.random.randn(len(dates))*0.02) * 10000, index=dates)
        returns_daily = np.log(price_1m).diff().dropna()
        
        return {
            'RV_1m_vol': RV_1m_vol,
            'RV_5m_vol': RV_5m_vol,
            'YZ_vol_daily': YZ_vol_daily,
            'price_1m': price_1m,
            'returns_daily': returns_daily
        }

# =============================================================================
# RV EXPLORER CLASS
# =============================================================================

class RVExplorer:
    """Interactive RV exploration tool"""
    
    def __init__(self, rv_series, name, freq):
        self.rv = rv_series.replace([np.inf, -np.inf], np.nan).dropna()
        self.name = name
        self.freq = freq
        self.log_rv = np.log(self.rv)
    
    def get_summary_stats(self):
        """Return summary statistics"""
        stats_dict = {
            'Observations': len(self.rv),
            'Start Date': self.rv.index.min().strftime('%Y-%m-%d'),
            'End Date': self.rv.index.max().strftime('%Y-%m-%d'),
            'Mean': f"{self.rv.mean():.6f}",
            'Std Dev': f"{self.rv.std():.6f}",
            'Min': f"{self.rv.min():.6f}",
            '25%': f"{self.rv.quantile(0.25):.6f}",
            '50%': f"{self.rv.median():.6f}",
            '75%': f"{self.rv.quantile(0.75):.6f}",
            'Max': f"{self.rv.max():.6f}",
            'Skewness': f"{stats.skew(self.rv):.4f}",
            'Kurtosis': f"{stats.kurtosis(self.rv):.4f}",
            'AR(1)': f"{self.rv.autocorr(lag=1):.4f}",
            'AR(5)': f"{self.rv.autocorr(lag=5):.4f}",
            'AR(22)': f"{self.rv.autocorr(lag=22):.4f}"
        }
        
        # ADF test
        adf = adfuller(self.rv.dropna(), autolag='AIC')
        stats_dict['ADF Statistic'] = f"{adf[0]:.4f}"
        stats_dict['ADF p-value'] = f"{adf[1]:.4f}"
        stats_dict['Stationary'] = "Yes" if adf[1] < 0.05 else "No"
        
        return stats_dict
    
    def plot_time_series(self, figsize=(10, 6)):
        """Create time series plot"""
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.rv.index, self.rv, linewidth=1, alpha=0.7, color='#1E88E5')
        
        # Add rolling means
        rolling_30 = self.rv.rolling(30).mean()
        rolling_90 = self.rv.rolling(90).mean()
        
        ax.plot(rolling_30.index, rolling_30, 'r-', linewidth=2, label='30-day MA', alpha=0.8)
        ax.plot(rolling_90.index, rolling_90, 'g-', linewidth=2, label='90-day MA', alpha=0.8)
        
        ax.set_title(f'{self.name} - Time Series', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Realized Volatility')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_distribution(self, figsize=(10, 6)):
        """Create distribution plots"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        axes[0].hist(self.rv, bins=50, edgecolor='black', alpha=0.7, color='#1E88E5')
        axes[0].axvline(self.rv.mean(), color='r', linestyle='--', 
                       linewidth=2, label=f'Mean: {self.rv.mean():.6f}')
        axes[0].axvline(self.rv.median(), color='g', linestyle='--', 
                       linewidth=2, label=f'Median: {self.rv.median():.6f}')
        axes[0].set_title('Distribution of RV', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('RV')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # QQ plot
        stats.probplot(self.log_rv.dropna(), dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Log RV)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_acf_pacf(self, lags=100, figsize=(12, 5)):
        """Create ACF and PACF plots"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        plot_acf(self.rv.dropna(), lags=lags, ax=axes[0], alpha=0.05)
        axes[0].set_title('Autocorrelation Function', fontsize=12, fontweight='bold')
        axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Half-life threshold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        plot_pacf(self.rv.dropna(), lags=40, ax=axes[1], alpha=0.05)
        axes[1].set_title('Partial Autocorrelation Function', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def analyze_clustering(self, threshold_percentile=75):
        """Analyze volatility clustering"""
        threshold = np.percentile(self.rv, threshold_percentile)
        high_vol = self.rv > threshold
        
        # Count clusters
        clusters = (high_vol != high_vol.shift()).cumsum()
        cluster_lengths = high_vol.groupby(clusters).sum()
        high_vol_clusters = cluster_lengths[cluster_lengths > 0]
        
        # Transition probabilities
        prob_hh = (high_vol.fillna(False) & high_vol.shift(1).fillna(False)).sum() / high_vol.fillna(False).sum() if high_vol.sum() > 0 else 0
        prob_ll = ((~high_vol.fillna(False)) & (~high_vol.shift(1).fillna(False))).sum() / (~high_vol.fillna(False)).sum() if (~high_vol).sum() > 0 else 0
        
        return {
            'threshold': threshold,
            'high_vol_days': high_vol.sum(),
            'high_vol_pct': high_vol.sum() / len(self.rv) * 100,
            'num_clusters': len(high_vol_clusters),
            'avg_cluster_length': high_vol_clusters.mean(),
            'max_cluster_length': high_vol_clusters.max(),
            'prob_hh': prob_hh,
            'prob_ll': prob_ll,
            'high_vol': high_vol
        }
    
    def plot_clustering(self, clustering_results, figsize=(12, 8)):
        """Plot volatility clustering"""
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Time series with threshold
        axes[0].plot(self.rv.index, self.rv, linewidth=0.8, alpha=0.7, color='#1E88E5')
        axes[0].fill_between(self.rv.index, 0, clustering_results['threshold'], 
                            alpha=0.2, color='green', label='Low volatility regime')
        axes[0].axhline(y=clustering_results['threshold'], color='r', linestyle='--', 
                       linewidth=2, label=f'75th percentile')
        axes[0].set_title('Volatility Clustering', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('RV')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Regime indicator
        axes[1].fill_between(self.rv.index, 0, 1, where=clustering_results['high_vol'], 
                            alpha=0.7, color='red', step='mid', label='High volatility')
        axes[1].set_title('High Volatility Regime Indicator', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Regime')
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(['Low', 'High'])
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig

# =============================================================================
# MAIN STREAMLIT APP
# =============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">📊 Bitcoin Realized Volatility Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive Analysis of Volatility Dynamics</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Bitcoin.svg/1200px-Bitcoin.svg.png", width=100)
        st.title("⚙️ Controls")
        
        # Load data
        with st.spinner("Loading data..."):
            data = load_and_prepare_data()
        
        # Date range selector
        st.subheader("📅 Date Range")
        
        # Get min and max dates (timezone-aware)
        min_date = min([s.index.min() for s in data.values() if isinstance(s, pd.Series)]).date()
        max_date = max([s.index.max() for s in data.values() if isinstance(s, pd.Series)]).date()
        
        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # RV type selector
        st.subheader("📈 RV Estimator")
        rv_type = st.selectbox(
            "Select RV Type",
            ["5-minute RV", "1-minute RV", "Yang-Zhang RV"]
        )
        
        # Analysis options
        st.subheader("🔍 Analysis Options")
        show_stats = st.checkbox("Summary Statistics", True)
        show_ts = st.checkbox("Time Series Plot", True)
        show_dist = st.checkbox("Distribution Analysis", True)
        show_acf = st.checkbox("Persistence Analysis", True)
        show_cluster = st.checkbox("Clustering Analysis", True)
        show_var = st.checkbox("Value at Risk (VaR)", True)
        
        # Period comparison
        st.subheader("📊 Period Comparison")
        compare_periods = st.checkbox("Compare Crisis Periods", True)
        
        # VaR settings
        st.subheader("📉 VaR Settings")
        conf = st.selectbox("Confidence level", [0.95, 0.99], index=1)
        var_alpha = 1 - conf
        var_dist = st.selectbox("Distribution", ["Normal", "Student-t"])
        var_df_t = st.slider("t degrees of freedom", 3, 30, 8) if var_dist == "Student-t" else 8
        var_use_mu = st.checkbox("Include rolling mean (μ)", value=False)
        var_position = st.selectbox("Position", ["Long", "Short"])
        
        # Download option
        st.subheader("💾 Export")
        if st.button("Download Current View"):
            st.info("Export feature - would save current analysis")
    
    # Main content area
    col1, col2, col3 = st.columns(3)
    
    # Select RV series based on user choice
    if rv_type == "1-minute RV":
        rv_series = data['RV_1m_vol']
        explorer = RVExplorer(rv_series, "1-minute RV", "1-min")
    elif rv_type == "5-minute RV":
        rv_series = data['RV_5m_vol']
        explorer = RVExplorer(rv_series, "5-minute RV", "5-min")
    else:
        rv_series = data['YZ_vol_daily']
        explorer = RVExplorer(rv_series, "Yang-Zhang RV", "Daily")
    
    # Filter by date range - FIXED TIMESTAMP ISSUE
    # Convert dates to timezone-aware timestamps
    start_timestamp = pd.Timestamp(start_date).tz_localize('UTC')
    end_timestamp = pd.Timestamp(end_date).tz_localize('UTC')
    
    mask = (rv_series.index >= start_timestamp) & (rv_series.index <= end_timestamp)
    rv_filtered = rv_series[mask]
    
    if len(rv_filtered) == 0:
        st.warning("No data available for selected date range. Showing full range.")
        rv_filtered = rv_series
        start_timestamp = rv_series.index.min()
        end_timestamp = rv_series.index.max()
    
    explorer_filtered = RVExplorer(rv_filtered, explorer.name, explorer.freq)
    
    # Quick metrics
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if len(rv_filtered) > 0:
            current_rv = rv_filtered.iloc[-1]
            prev_rv = rv_filtered.iloc[-2] if len(rv_filtered) > 1 else current_rv
            pct_change = ((current_rv - prev_rv) / prev_rv * 100) if prev_rv != 0 else 0
            st.metric("Current RV", f"{current_rv:.6f}", f"{pct_change:.1f}%")
        else:
            st.metric("Current RV", "N/A", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if len(rv_filtered) > 0:
            st.metric("Average RV", f"{rv_filtered.mean():.6f}", f"±{rv_filtered.std():.6f}")
        else:
            st.metric("Average RV", "N/A", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Data Points", f"{len(rv_filtered):,}", explorer.name)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary Statistics
    if show_stats:
        st.subheader("📊 Summary Statistics")
        stats_dict = explorer_filtered.get_summary_stats()
        
        # Display in columns
        cols = st.columns(4)
        stat_items = list(stats_dict.items())
        for i, (key, value) in enumerate(stat_items):
            with cols[i % 4]:
                st.markdown(f"**{key}:** {value}")
    
    # Time Series Plot
    if show_ts:
        st.subheader("📈 Time Series Analysis")
        fig = explorer_filtered.plot_time_series()
        st.pyplot(fig)
        plt.close()
    
    # Distribution Analysis
    if show_dist:
        st.subheader("📊 Distribution Analysis")
        fig = explorer_filtered.plot_distribution()
        st.pyplot(fig)
        plt.close()
    
    # Persistence Analysis
    if show_acf:
        st.subheader("🔄 Persistence Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            lags = st.slider("Number of lags for ACF", 20, 200, 100, key='acf_lags')
        
        fig = explorer_filtered.plot_acf_pacf(lags=lags)
        st.pyplot(fig)
        plt.close()
        
        # Half-life calculation
        acf_values = acf(explorer_filtered.rv.dropna(), nlags=lags)
        try:
            half_life = np.where(acf_values < 0.5)[0][0]
            st.markdown(f'<div class="insight-box">💡 **Insight:** Volatility shocks have a half-life of approximately **{half_life} days**. This means it takes about {half_life} days for a volatility spike to decay by 50%.</div>', unsafe_allow_html=True)
        except:
            st.markdown(f'<div class="warning-box">⚠️ **Note:** Volatility is extremely persistent - half-life exceeds {lags} days.</div>', unsafe_allow_html=True)
    
    # Clustering Analysis
    if show_cluster:
        st.subheader("🔴 Volatility Clustering Analysis")
        
        threshold = st.slider("High volatility threshold (percentile)", 50, 95, 75, key='cluster_thresh')
        cluster_results = explorer_filtered.analyze_clustering(threshold_percentile=threshold)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("High Vol Days", f"{cluster_results['high_vol_days']}", 
                     f"{cluster_results['high_vol_pct']:.1f}%")
        with col2:
            st.metric("Number of Clusters", f"{cluster_results['num_clusters']}")
        with col3:
            st.metric("Avg Cluster Length", f"{cluster_results['avg_cluster_length']:.1f} days")
        with col4:
            st.metric("Max Cluster", f"{cluster_results['max_cluster_length']:.0f} days")
        
        # Transition probabilities
        col1, col2 = st.columns(2)
        with col1:
            st.metric("P(High → High)", f"{cluster_results['prob_hh']:.3f}")
        with col2:
            st.metric("P(Low → Low)", f"{cluster_results['prob_ll']:.3f}")
        
        # Insight
        if cluster_results['prob_hh'] > 0.7:
            st.markdown(f'<div class="insight-box">💡 **Strong clustering detected:** When volatility is high, there\'s a {cluster_results["prob_hh"]*100:.1f}% chance it stays high tomorrow. This is characteristic of turbulent market periods.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="insight-box">💡 **Moderate clustering:** Volatility shows some persistence but isn\'t extremely clustered.</div>', unsafe_allow_html=True)
        
        # Plot clustering
        fig = explorer_filtered.plot_clustering(cluster_results)
        st.pyplot(fig)
        plt.close()
    
    # Period Comparison
    if compare_periods:
        st.subheader("📅 Crisis Period Analysis")
        
        # Define crisis periods with timezone awareness
        crisis_periods = [
            ('2017-12-01', '2018-02-28', '2017-18 Bubble Burst'),
            ('2020-03-01', '2020-04-30', 'COVID-19 Crash'),
            ('2021-04-01', '2021-07-31', '2021 China Crackdown'),
            ('2021-11-01', '2022-01-31', '2021-22 Peak Correction'),
            ('2022-05-01', '2022-06-30', 'Terra/Luna Collapse'),
            ('2022-11-01', '2022-12-31', 'FTX Collapse'),
        ]
        
        # Create comparison DataFrame
        comparison_data = []
        
        # Normal period (all data not in crisis)
        crisis_mask = pd.Series(False, index=rv_filtered.index)
        for start, end, name in crisis_periods:
            start_ts = pd.Timestamp(start).tz_localize('UTC')
            end_ts = pd.Timestamp(end).tz_localize('UTC')
            mask = (rv_filtered.index >= start_ts) & (rv_filtered.index <= end_ts)
            crisis_mask |= mask
        
        normal_rv = rv_filtered[~crisis_mask]
        if len(normal_rv) > 0:
            comparison_data.append({
                'Period': 'Normal',
                'Obs': len(normal_rv),
                'Mean RV': normal_rv.mean(),
                'Std RV': normal_rv.std(),
                'Max RV': normal_rv.max(),
                'AR(1)': normal_rv.autocorr(lag=1)
            })
        
        # Crisis periods
        for start, end, name in crisis_periods:
            start_ts = pd.Timestamp(start).tz_localize('UTC')
            end_ts = pd.Timestamp(end).tz_localize('UTC')
            crisis_rv = rv_filtered[(rv_filtered.index >= start_ts) & (rv_filtered.index <= end_ts)]
            if len(crisis_rv) > 0:
                comparison_data.append({
                    'Period': name,
                    'Obs': len(crisis_rv),
                    'Mean RV': crisis_rv.mean(),
                    'Std RV': crisis_rv.std(),
                    'Max RV': crisis_rv.max(),
                    'AR(1)': crisis_rv.autocorr(lag=1)
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Format for display
            display_df = comparison_df.copy()
            for col in ['Mean RV', 'Std RV', 'Max RV']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}")
            display_df['AR(1)'] = display_df['AR(1)'].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Visualization
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Mean comparison
            axes[0].bar(comparison_df['Period'], comparison_df['Mean RV'], alpha=0.7, color='#1E88E5')
            axes[0].set_title('Mean RV by Period', fontweight='bold')
            axes[0].set_ylabel('Mean RV')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # Persistence comparison
            axes[1].bar(comparison_df['Period'], comparison_df['AR(1)'], alpha=0.7, color='#FF9800')
            axes[1].set_title('Persistence (AR1) by Period', fontweight='bold')
            axes[1].set_ylabel('AR(1) Coefficient')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3, axis='y')
            axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Business insight
            st.markdown('<div class="insight-box">💡 **Trading Insight:** During crisis periods, volatility is typically 2-3x higher and more persistent. This suggests:'
                       '\n1. Position sizing should be reduced during crises'
                       '\n2. Stop-losses need to be wider'
                       '\n3. Mean-reversion strategies may fail as volatility clusters'
                       '\n4. Option premiums (implied volatility) should be higher</div>', unsafe_allow_html=True)
    
    # Estimator Comparison
    if rv_type != "Yang-Zhang RV":
        st.subheader("📊 Estimator Comparison")
        
        # Compare with Yang-Zhang
        yz_series = data['YZ_vol_daily']
        common_idx = rv_filtered.index.intersection(yz_series.index)
        
        if len(common_idx) > 0:
            rv_common = rv_filtered.loc[common_idx]
            yz_common = yz_series.loc[common_idx]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Correlation", f"{rv_common.corr(yz_common):.4f}")
            with col2:
                efficiency_gain = (rv_common.std() - yz_common.std()) / rv_common.std() * 100
                st.metric("Efficiency Gain", f"{efficiency_gain:.2f}%", 
                         "Yang-Zhang more efficient" if efficiency_gain > 0 else "Standard more efficient")
            with col3:
                noise_ratio = (rv_common.mean() - yz_common.mean()) / rv_common.mean() * 100
                st.metric("Microstructure Noise", f"{abs(noise_ratio):.2f}%")
            
            # Scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(rv_common, yz_common, alpha=0.5, s=20)
            ax.plot([rv_common.min(), rv_common.max()], 
                   [rv_common.min(), rv_common.max()], 'r--', label='45° line')
            ax.set_xlabel(f'{rv_type}')
            ax.set_ylabel('Yang-Zhang RV')
            ax.set_title(f'{rv_type} vs Yang-Zhang (Corr: {rv_common.corr(yz_common):.3f})', 
                        fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
    
    # Value at Risk (VaR)
    if show_var:
        st.subheader("📉 Value at Risk (VaR)")
        
        # Use the selected RV series (filtered to date range) as the volatility estimate
        # and retrieve daily returns aligned to the same period
        returns_daily = data['returns_daily']
        
        # Filter returns to the selected date range
        ret_mask = (returns_daily.index >= start_timestamp) & (returns_daily.index <= end_timestamp)
        returns_filtered = returns_daily[ret_mask]
        
        # Use rv_filtered as sigma estimate
        sigma_hat = rv_filtered.copy()
        
        # Compute VaR
        var_series = compute_var(
            returns=returns_filtered,
            sigma=sigma_hat,
            alpha=var_alpha,
            dist=var_dist,
            df_t=var_df_t,
            use_mu=var_use_mu
        )
        
        # Flip for short position
        if var_position == "Short":
            var_series = -var_series
        
        # Align returns and VaR
        r_aligned = returns_filtered.reindex(var_series.index).dropna()
        v_aligned = var_series.reindex(r_aligned.index).dropna()
        
        # Exceptions
        exceptions = (r_aligned < v_aligned) if var_position == "Long" else (r_aligned > v_aligned)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Observations", f"{len(exceptions):,}")
        col2.metric("Exceptions", f"{int(exceptions.sum()):,}")
        col3.metric("Expected", f"{var_alpha * len(exceptions):.1f}")
        
        # Exception rate insight
        if len(exceptions) > 0:
            actual_rate = exceptions.sum() / len(exceptions)
            expected_rate = var_alpha
            if actual_rate > expected_rate * 1.5:
                st.markdown(
                    f'<div class="warning-box">⚠️ <b>Backtesting Warning:</b> Actual exception rate '
                    f'({actual_rate*100:.2f}%) exceeds expected ({expected_rate*100:.2f}%). '
                    f'The model may be <b>underestimating</b> risk.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="insight-box">✅ <b>Backtesting OK:</b> Actual exception rate '
                    f'({actual_rate*100:.2f}%) is close to expected ({expected_rate*100:.2f}%).</div>',
                    unsafe_allow_html=True
                )
        
        st.line_chart(pd.DataFrame({"Return": r_aligned, "VaR": v_aligned}))
        
        st.dataframe(
            pd.DataFrame({
                "Return": r_aligned,
                "VaR": v_aligned,
                "Exception": exceptions
            }).tail(20),
            use_container_width=True
        )
    
    # Footer with business insights
    st.markdown("---")
    st.subheader("💼 Business Insights for Traders")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📈 Volatility Dynamics:**
        - **Persistence:** High volatility tends to cluster (like storms)
        - **Mean Reversion:** Volatility eventually returns to normal levels
        - **Asymmetry:** Negative returns often increase volatility more than positive returns
        
        **🛠️ Trading Applications:**
        - **Position Sizing:** Reduce size when RV > 2× average
        - **Stop-Loss Placement:** Set wider stops during high RV periods
        - **Option Pricing:** Compare realized vs implied volatility
        """)
    
    with col2:
        st.markdown("""
        **⚠️ Risk Management:**
        - **VaR Models:** Use t-distribution (not normal) for fat tails
        - **Stress Testing:** Test portfolio under crisis RV levels
        - **Hedging:** More expensive during high RV clusters
        
        **🎯 Strategy Implications:**
        - **Mean Reversion:** Works better in normal RV periods
        - **Trend Following:** Can thrive during high RV clusters
        - **Market Making:** Widen spreads when RV spikes
        """)
    
    # Export option
    st.markdown("---")
    if st.button("📥 Export Analysis Results"):
        # Create export DataFrame
        export_df = pd.DataFrame({
            'Date': rv_filtered.index,
            'RV': rv_filtered.values,
            'Log_RV': np.log(rv_filtered).values
        })
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{rv_type.replace(' ', '_')}_analysis.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
    
