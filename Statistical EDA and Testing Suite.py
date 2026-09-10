import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller
from scipy.stats import anderson
from matplotlib.ticker import MaxNLocator
import pymannkendall as mk

# ==============================================================================
# GLOBAL STYLING & PARAMETER DEFINITIONS
# ==============================================================================
plt.rcParams['font.family'] = 'DejaVu Sans'  # Set to 'Liberation Serif' if installed locally
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
DPI = 300

PARAM_COLS = [
    'PM₂.₅ (µg/m³)', 'NO (µg/m³)', 'NO₂ (µg/m³)', 'NOx (µg/m³)',
    'SO₂ (µg/m³)', 'CO (mg/m³)', 'O₃ (µg/m³)', 'Benzene (µg/m³)',
    'Toluene (µg/m³)', 't2m (Kelvin)', 'd2m (Kelvin)', 'sp (Pascals)',
    'tp (m)', 'i10fg(m/s)', 'r (%)', 'ws (m/s)', 'wd (°)'
]

Y_LABELS = {
    'PM₂.₅ (µg/m³)': r'PM$_{2.5}$ ($\mu$g/m$^3$)',
    'NO (µg/m³)': r'NO ($\mu$g/m$^3$)',
    'NO₂ (µg/m³)': r'NO$_2$ ($\mu$g/m$^3$)',
    'NOx (µg/m³)': r'NO$_x$ ($\mu$g/m$^3$)',
    'SO₂ (µg/m³)': r'SO$_2$ ($\mu$g/m$^3$)',
    'CO (mg/m³)': r'CO (mg/m$^3$)',
    'O₃ (µg/m³)': r'O$_3$ ($\mu$g/m$^3$)',
    'Benzene (µg/m³)': r'Benzene ($\mu$g/m$^3$)',
    'Toluene (µg/m³)': r'Toluene ($\mu$g/m$^3$)',
    't2m (Kelvin)': 'Air Temperature (K)',
    'd2m (Kelvin)': 'Dew Point (K)',
    'sp (Pascals)': 'Surface Pressure (Pa)',
    'tp (m)': 'Total Precipitation (m)',
    'i10fg(m/s)': '10 m Wind Gust (m/s)',
    'r (%)': 'Relative Humidity (%)',
    'ws (m/s)': 'Wind Speed (m/s)',
    'wd (°)': 'Wind Direction (°)'
}

PALETTE = [
    "#4477AA", "#EE7733", "#CC3311", "#AA4499", "#228833", 
    "#997700", "#66CCEE", "#FF006E", "#BBBB22", "#009988", 
    "#CC79A7", "#332288", "#00ACC1", "#6A3D9A", "#C2185B", 
    "#88CC44", "#DDCC77"
]
COLOR_MAP = dict(zip(PARAM_COLS, PALETTE))

# ==============================================================================
# DATA INGESTION & PREPARATION
# ==============================================================================
def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Reads Excel dataset, parses datetime, and ensures numeric formatting."""
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    
    date_col = next((c for c in df.columns if c.lower() == 'from date'), None)
    if not date_col:
        raise KeyError("Dataset must contain a 'From Date' column.")
        
    df['From Date'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['From Date']).sort_values('From Date').reset_index(drop=True)
    df['Year'] = df['From Date'].dt.year

    for col in PARAM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

# ==============================================================================
# 1. ANNUAL MEAN BAR PLOTS
# ==============================================================================
def run_annual_mean_test(df: pd.DataFrame, output_dir: str):
    """Calculates yearly averages and saves publication bar plots."""
    os.makedirs(output_dir, exist_ok=True)
    annual = df.groupby('Year')[PARAM_COLS].mean().reset_index()

    for col in PARAM_COLS:
        if col not in annual.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
        ax.bar(annual['Year'], annual[col], color=COLOR_MAP.get(col, '#4477AA'), 
               edgecolor='black', linewidth=1.5, width=0.6)
        
        ax.set_title(f"Annual Mean: {col}", fontsize=14, pad=12)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel(Y_LABELS.get(col, col), fontsize=12)
        ax.set_xticks(annual['Year'])
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        clean_name = re.sub(r'[^a-zA-Z0-9]+', '_', col)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"AnnualMean_{clean_name}.png"))
        plt.close()
    print(f"Annual Mean plots saved to: {output_dir}")

# ==============================================================================
# 2. LINEARITY & TREND REGRESSION
# ==============================================================================
def run_linearity_test(df: pd.DataFrame, output_dir: str):
    """Evaluates linear trend strength over time via Linear Regression."""
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- Linearity Test ($R^2$ Evaluation) ---")

    for col in PARAM_COLS:
        subset = df[['From Date', col]].dropna().sort_values('From Date')
        if len(subset) < 3:
            continue
            
        X = subset['From Date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        y = subset[col].values.reshape(-1, 1)

        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        r2 = r2_score(y, y_pred)
        
        status = "Strong" if r2 > 0.7 else "Moderate" if r2 > 0.3 else "Weak/Non-linear"
        print(f"{col:20s} | R² = {r2:7.4f} ({status})")

        fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
        ax.scatter(subset['From Date'], y, color=COLOR_MAP.get(col, '#4477AA'), 
                   s=15, alpha=0.6, label='Observations')
        ax.plot(subset['From Date'], y_pred, color='red', linewidth=2, label=f'Fit ($R^2={r2:.3f}$)')
        
        ax.set_title(f"Linearity Analysis: {col}", fontsize=14, pad=12)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(Y_LABELS.get(col, col), fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        clean_name = re.sub(r'[^a-zA-Z0-9]+', '_', col)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Linearity_{clean_name}.png"))
        plt.close()

# ==============================================================================
# 3. STATIONARITY TEST (AUGMENTED DICKEY-FULLER)
# ==============================================================================
def run_stationarity_test(df: pd.DataFrame, output_dir: str, window: int = 30):
    """Computes ADF test statistics and generates rolling mean/std plots."""
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- Augmented Dickey-Fuller (ADF) Test ---")
    daily_df = df.set_index('From Date')[PARAM_COLS].resample('D').mean().dropna()

    for col in PARAM_COLS:
        series = daily_df[col].dropna()
        if len(series) < 10:
            continue
            
        stat, p_val, _, _, crit, _ = adfuller(series)
        is_stationary = p_val < 0.05
        result_str = "Stationary (Reject H0)" if is_stationary else "Non-Stationary (Fail to Reject H0)"
        print(f"{col:20s} | ADF: {stat:8.4f} | p-value: {p_val:6.4f} | Status: {result_str}")

        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()

        fig, ax = plt.subplots(figsize=(11, 6), dpi=DPI)
        ax.plot(series.index, series, color=COLOR_MAP.get(col, '#4477AA'), alpha=0.35, label='Daily Average')
        ax.plot(rolling_mean.index, rolling_mean, color='black', linewidth=1.8, label=f'{window}-Day Rolling Mean')
        ax.plot(rolling_std.index, rolling_std, color='crimson', linestyle='--', linewidth=1.5, label=f'{window}-Day Rolling Std')
        
        ax.set_title(f"Stationarity Test (ADF p={p_val:.4f}): {col}", fontsize=13)
        ax.set_ylabel(Y_LABELS.get(col, col), fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.5)

        clean_name = re.sub(r'[^a-zA-Z0-9]+', '_', col)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"ADF_{clean_name}.png"))
        plt.close()

# ==============================================================================
# 4. NORMALITY TEST (ANDERSON-DARLING)
# ==============================================================================
def run_normality_test(df: pd.DataFrame, output_dir: str):
    """Executes Anderson-Darling test with Freedman-Diaconis histogram & KDE."""
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- Anderson-Darling Normality Test ---")

    for col in PARAM_COLS:
        data = df[col].dropna()
        if len(data) < 8:
            continue
            
        res = anderson(data, dist='norm')
        sig_5_idx = list(res.significance_level).index(5.0)
        h0_rejected = res.statistic > res.critical_values[sig_5_idx]
        print(f"{col:20s} | A²: {res.statistic:8.3f} | 5% Crit: {res.critical_values[sig_5_idx]:5.3f} | Rejected H0: {h0_rejected}")

        fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
        sns.histplot(data, kde=True, color=COLOR_MAP.get(col, '#4477AA'), 
                     edgecolor='black', alpha=0.7, ax=ax)
        ax.set_title(f"Distribution & Normality: {col} (A² = {res.statistic:.2f})", fontsize=13)
        ax.set_xlabel(Y_LABELS.get(col, col), fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.5)

        clean_name = re.sub(r'[^a-zA-Z0-9]+', '_', col)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Normality_{clean_name}.png"))
        plt.close()

# ==============================================================================
# 5. SPEARMAN CORRELATION HEATMAP
# ==============================================================================
def run_spearman_heatmap(df: pd.DataFrame, output_dir: str):
    """Generates a publication-grade Spearman rank correlation heatmap."""
    os.makedirs(output_dir, exist_ok=True)
    num_df = df[PARAM_COLS].select_dtypes(include='number').dropna()
    corr = num_df.corr(method='spearman')

    fig, ax = plt.subplots(figsize=(13, 11), dpi=DPI)
    labels = [Y_LABELS.get(c, c) for c in corr.columns]
    
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5,
                linecolor='black', xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={'shrink': 0.8})
    
    ax.set_title("Spearman Rank Correlation Matrix", fontsize=15, pad=18)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Spearman_Correlation_Matrix.png"))
    plt.close()
    print(f"Spearman Heatmap saved to: {output_dir}")

# ==============================================================================
# 6. MANN-KENDALL TREND TEST
# ==============================================================================
def run_mann_kendall_test(df: pd.DataFrame, output_dir: str):
    """Performs Mann-Kendall non-parametric monotonic trend test."""
    os.makedirs(output_dir, exist_ok=True)
    daily_df = df.set_index('From Date')[PARAM_COLS].resample('D').mean().dropna()
    records = []

    for col in PARAM_COLS:
        series = daily_df[col].dropna()
        if len(series) < 5:
            continue
        res = mk.original_test(series)
        val = 1 if res.trend == 'increasing' else -1 if res.trend == 'decreasing' else 0
        records.append({'Parameter': col, 'Trend': res.trend, 'Score': val, 'p-val': res.p, 'Slope': res.slope})

    res_df = pd.DataFrame(records)
    print("\n--- Mann-Kendall Trend Results ---")
    print(res_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)
    colors = [COLOR_MAP.get(p, 'skyblue') for p in res_df['Parameter']]
    ax.bar(res_df['Parameter'], res_df['Score'], color=colors, edgecolor='black', linewidth=1.2)
    ax.axhline(0, color='black', linewidth=2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['Decreasing (-1)', 'No Trend (0)', 'Increasing (+1)'], fontsize=11)
    ax.set_xticklabels([Y_LABELS.get(p, p) for p in res_df['Parameter']], rotation=45, ha='right')
    ax.set_title("Mann-Kendall Monotonic Trend Direction", fontsize=14, pad=12)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Mann_Kendall_Trend_Test.png"))
    plt.close()
    res_df.to_csv(os.path.join(output_dir, "Mann_Kendall_Results.csv"), index=False)

# ==============================================================================
# MAIN EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    # Update with your local path or Google Colab path
    DATA_PATH = "chennai.xlsx"
    OUTPUT_BASE = "Output_Statistical_Tests"

    if os.path.exists(DATA_PATH):
        df_clean = load_and_clean_data(DATA_PATH)
        run_annual_mean_test(df_clean, os.path.join(OUTPUT_BASE, "Annual_Mean"))
        run_linearity_test(df_clean, os.path.join(OUTPUT_BASE, "Linearity"))
        run_stationarity_test(df_clean, os.path.join(OUTPUT_BASE, "Stationarity"))
        run_normality_test(df_clean, os.path.join(OUTPUT_BASE, "Normality"))
        run_spearman_heatmap(df_clean, os.path.join(OUTPUT_BASE, "Correlation"))
        run_mann_kendall_test(df_clean, os.path.join(OUTPUT_BASE, "Mann_Kendall"))
    else:
        print(f"Data file not found at '{DATA_PATH}'. Place your dataset in the root folder to execute.")