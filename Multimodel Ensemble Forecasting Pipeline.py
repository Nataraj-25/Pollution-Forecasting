import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter1d

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ==============================================================================
# PIPELINE CONFIGURATION
# ==============================================================================
AIR_POLLUTANTS = [
    'PM₂.₅ (µg/m³)', 'NO (µg/m³)', 'NO₂ (µg/m³)', 'NOx (µg/m³)',
    'SO₂ (µg/m³)', 'CO (mg/m³)', 'O₃ (µg/m³)', 'Benzene (µg/m³)',
    'Toluene (µg/m³)'
]

METEOROLOGICAL_PARAMS = [
    't2m (Kelvin)', 'd2m (Kelvin)', 'sp (Pascals)', 'tp (m)',
    'r (%)', 'i10fg(m/s)', 'ws (m/s)', 'wd (°)'
]

ALL_FEATURES = AIR_POLLUTANTS + METEOROLOGICAL_PARAMS

POLLUTANT_COLORS = {
    'PM₂.₅ (µg/m³)': '#4477AA',
    'NO (µg/m³)': '#EE7733',
    'NO₂ (µg/m³)': '#CC3311',
    'NOx (µg/m³)': '#AA4499',
    'SO₂ (µg/m³)': '#228833',
    'CO (mg/m³)': '#997700',
    'O₃ (µg/m³)': '#66CCEE',
    'Benzene (µg/m³)': '#FF006E',
    'Toluene (µg/m³)': '#BBBB22'
}

# ==============================================================================
# DATA LOADER & TEMPORAL PARTITION
# ==============================================================================
def prepare_air_quality_data(filepath: str):
    """Loads and splits the time-series into 2020-2022 train and 2023-2025 test."""
    data = pd.read_excel(filepath)
    data.columns = data.columns.str.strip()

    date_col = next((c for c in data.columns if c.lower() == 'from date'), None)
    data['From Date'] = pd.to_datetime(data[date_col], errors='coerce', dayfirst=True)
    data = data.dropna(subset=['From Date']).set_index('From Date').sort_index()

    # Chronological partition
    train_data = data[data.index.year < 2023].copy()
    test_data = data[data.index.year.isin([2023, 2024, 2025])].copy()
    
    return train_data, test_data

# ==============================================================================
# MODEL TRAINING & ENSEMBLE FORECASTING
# ==============================================================================
def train_and_evaluate_ensemble(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str):
    """Fits MLP, XGBoost, and GBR across a 5-fold TimeSeriesSplit, then averages."""
    os.makedirs(output_dir, exist_ok=True)
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Model Base Estimators
    mlp = MLPRegressor(hidden_layer_sizes=(32,), alpha=0.05, activation='relu',
                       solver='adam', max_iter=1000, early_stopping=True,
                       n_iter_no_change=15, random_state=42)
    xgb = XGBRegressor(n_estimators=80, learning_rate=0.05, max_depth=3,
                       subsample=0.75, colsample_bytree=0.75, random_state=42)
    gbr = GradientBoostingRegressor(n_estimators=80, learning_rate=0.05, max_depth=3,
                                    subsample=0.75, random_state=42)

    predictions = pd.DataFrame(index=test_df.index)
    metrics = {}

    print("\n=======================================================")
    print("Beginning Multi-Model Training & Evaluation")
    print("=======================================================")

    for pollutant in AIR_POLLUTANTS:
        X_train = train_df[ALL_FEATURES]
        y_train = train_df[pollutant]
        X_test = test_df[ALL_FEATURES]
        y_test = test_df[pollutant]

        # Standard scaling specifically for MLP convergence
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        mlp_folds, xgb_folds, gbr_folds = [], [], []

        for train_idx, _ in tscv.split(X_train):
            # Fit and predict MLP
            m_mlp = clone(mlp).fit(X_train_scaled[train_idx], y_train.iloc[train_idx])
            mlp_folds.append(m_mlp.predict(X_test_scaled))

            # Fit and predict XGBoost
            m_xgb = clone(xgb).fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            xgb_folds.append(m_xgb.predict(X_test))

            # Fit and predict Gradient Boosting
            m_gbr = clone(gbr).fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            gbr_folds.append(m_gbr.predict(X_test))

        # Average folds then equal-weight ensemble
        p_mlp = np.mean(mlp_folds, axis=0)
        p_xgb = np.mean(xgb_folds, axis=0)
        p_gbr = np.mean(gbr_folds, axis=0)
        final_ensemble = (p_mlp + p_xgb + p_gbr) / 3.0
        
        predictions[pollutant] = final_ensemble

        # Calculate metrics
        mae = mean_absolute_error(y_test, final_ensemble)
        mse = mean_squared_error(y_test, final_ensemble)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, final_ensemble)
        metrics[pollutant] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

        print(f"Target: {pollutant:18s} | R²: {r2:6.4f} | RMSE: {rmse:7.4f} | MAE: {mae:7.4f}")

    metrics_df = pd.DataFrame(metrics).T
    print("\n--- Final Performance Table ---")
    print(metrics_df)

    # Save outputs
    metrics_df.to_csv(os.path.join(output_dir, "Ensemble_Performance_Metrics.csv"))
    
    comparison_df = pd.concat([test_df[AIR_POLLUTANTS], predictions.add_suffix('_Predicted')], axis=1)
    comparison_df.to_excel(os.path.join(output_dir, "Ensemble_Predictions.xlsx"))
    
    return test_df, predictions

# ==============================================================================
# SMOOTHED TIME-SERIES VISUALIZATION
# ==============================================================================
def plot_actual_vs_predicted(actual_df: pd.DataFrame, pred_df: pd.DataFrame, 
                             output_dir: str, start_date: str = "2024-01-01"):
    """Applies Gaussian smoothing and plots actual vs predicted concentrations."""
    os.makedirs(output_dir, exist_ok=True)
    mask = actual_df.index >= pd.Timestamp(start_date)
    dates = actual_df.loc[mask].index

    for pollutant in AIR_POLLUTANTS:
        act = actual_df.loc[mask, pollutant].values
        prd = pred_df.loc[mask, pollutant].values

        # 1D Gaussian filter smoothing (sigma=2)
        act_smooth = gaussian_filter1d(act, sigma=2)
        prd_smooth = gaussian_filter1d(prd, sigma=2)

        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        ax.plot(dates, act_smooth, color='black', linewidth=1.8, label='Actual')
        ax.plot(dates, prd_smooth, color=POLLUTANT_COLORS.get(pollutant, 'crimson'),
                linewidth=1.8, label='Ensemble Forecast')

        ax.set_title(f"Air Quality Prediction: {pollutant} (from {start_date})", fontsize=14, pad=12)
        ax.set_xlabel("Time (Month-Year)", fontsize=11)
        ax.set_ylabel(pollutant, fontsize=11)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.xticks(rotation=45, ha='right')

        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper right')

        clean_name = re.sub(r'[^a-zA-Z0-9]+', '_', pollutant)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Forecast_{clean_name}.png"))
        plt.close()

    print(f"Publication-style plots generated in: {output_dir}")

# ==============================================================================
# MAIN EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    DATASET_PATH = "chennai.xlsx"
    OUT_DIR = "Output_Ensemble_Forecast"

    if os.path.exists(DATASET_PATH):
        train_df, test_df = prepare_air_quality_data(DATASET_PATH)
        actuals, preds = train_and_evaluate_ensemble(train_df, test_df, OUT_DIR)
        plot_actual_vs_predicted(actuals, preds, os.path.join(OUT_DIR, "Plots"), start_date="2024-01-01")
    else:
        print(f"Dataset '{DATASET_PATH}' not found in current directory. Please place it here to run.")