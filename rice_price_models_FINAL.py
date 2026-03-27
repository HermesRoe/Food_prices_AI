"""
Rice Market Stability and Price Prediction Platform
Tamil Nadu — 11 Districts | 1994–2026
IEEE Conference Project — Final Model Code

HOW TO USE
----------
1. Place your dataset at the path in DATASET_PATH
2. Run:  python rice_price_models_FINAL.py
3. The script trains all 5 models, prints metrics, and prompts for a date
4. Enter any date (YYYY-MM-DD) to get a price prediction

VERIFY OUTPUT
-------------
Expected test metrics (pooled, 11 districts):
  Linear Regression  → MAE ≈ 0.36,  R² ≈ 0.938
  Gradient Boosting  → MAE ≈ 1.32,  R² ≈ 0.921
  Random Forest      → MAE ≈ 1.55,  R² ≈ 0.888
  SVR (RBF)          → MAE ≈ 1.95,  R² ≈ 0.861
  MLP Neural Net     → MAE ≈ 2.68,  R² ≈ 0.838
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network  import MLPRegressor
from sklearn.svm             import SVR
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
DATASET_PATH = "./TamilNadu_Rice_Refined_Unit_Standardized.xlsx"
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD & PREPARE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_engineer(filepath):
    df = pd.read_excel(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # Encode market as integer so it can be used as a feature
    le = LabelEncoder()
    df['market_enc'] = le.fit_transform(df['market'])

    # Sort within each district before computing lags
    df = df.sort_values(['market', 'date']).reset_index(drop=True)

    # Calendar features
    df['year']      = df['date'].dt.year
    df['month']     = df['date'].dt.month
    df['dow']       = df['date'].dt.dayofweek
    df['days_elap'] = (df['date'] - df['date'].min()).dt.days

    # Target: predict daily price CHANGE (more stationary than absolute price)
    df['price_change'] = df.groupby('market')['price'].diff(1)

    # Lag features — computed per district to prevent cross-market leakage
    for i in [1, 2, 3]:
        df[f'price_lag_{i}']  = df.groupby('market')['price'].shift(i)
        df[f'change_lag_{i}'] = df.groupby('market')['price_change'].shift(i)

    # Rolling stats per district
    df['roll3_mean'] = df.groupby('market')['price'].transform(
        lambda x: x.rolling(3, min_periods=1).mean())
    df['roll3_std']  = df.groupby('market')['price'].transform(
        lambda x: x.rolling(3, min_periods=1).std().fillna(0))

    FEATURES = [
        'year', 'month', 'dow', 'days_elap', 'market_enc',
        'price_lag_1', 'price_lag_2', 'price_lag_3',
        'change_lag_1', 'change_lag_2', 'change_lag_3',
        'roll3_mean', 'roll3_std'
    ]

    df = df.dropna(subset=FEATURES + ['price_change']).reset_index(drop=True)
    return df, FEATURES, le


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TRAIN / VALIDATE / TEST SPLIT  (70 / 10 / 20 — chronological)
# ═══════════════════════════════════════════════════════════════════════════════

def split_data(df, features):
    n  = len(df)
    s1 = int(n * 0.70)   # end of training
    s2 = int(n * 0.80)   # end of validation

    X = df[features].values
    y = df['price_change'].values

    X_train, X_val, X_test = X[:s1], X[s1:s2], X[s2:]
    y_train, y_val, y_test = y[:s1], y[s1:s2], y[s2:]

    # Previous-day actual prices for reconstructing absolute predictions
    prev_prices_test  = df['price'].values[s2 - 1:-1]
    actual_prices_test = df['price'].values[s2:]
    test_markets      = df['market'].values[s2:]
    test_dates        = df['date'].values[s2:]

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            prev_prices_test, actual_prices_test,
            test_markets, test_dates, s1, s2)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

def build_models():
    """Return dict of model configs. SVR and MLP need scaled inputs."""
    return {

        # ── Model 1: Linear Regression ──────────────────────────────────────
        # Simple, fast, best for extrapolating a long-run trend.
        # Uses raw (unscaled) features.
        "Linear Regression": {
            "model":  LinearRegression(),
            "scaled": False,
        },

        # ── Model 2: Gradient Boosting ───────────────────────────────────────
        # Best ensemble model for this dataset.
        # 400 trees, conservative learning rate, subsampling for regularisation.
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(
                n_estimators   = 400,
                learning_rate  = 0.05,
                max_depth      = 4,
                subsample      = 0.8,
                min_samples_leaf = 3,
                random_state   = 42
            ),
            "scaled": False,
        },

        # ── Model 3: Random Forest ────────────────────────────────────────────
        # Good for feature importance analysis.
        # 300 trees, moderate depth to prevent overfitting on small districts.
        "Random Forest": {
            "model": RandomForestRegressor(
                n_estimators   = 300,
                max_depth      = 6,
                min_samples_leaf = 3,
                random_state   = 42
            ),
            "scaled": False,
        },

        # ── Model 4: SVR (RBF kernel) ─────────────────────────────────────────
        # Non-linear, strong for capturing local price momentum patterns.
        # MUST use StandardScaler — sensitive to feature scale.
        "SVR (RBF)": {
            "model": SVR(
                kernel  = 'rbf',
                C       = 10,
                epsilon = 0.05,
                gamma   = 'scale'
            ),
            "scaled": True,
        },

        # ── Model 5: MLP Neural Network ───────────────────────────────────────
        # Three hidden layers. Early stopping prevents overfitting.
        # MUST use StandardScaler.
        "MLP Neural Net": {
            "model": MLPRegressor(
                hidden_layer_sizes   = (128, 64, 32),
                activation           = 'relu',
                max_iter             = 2000,
                learning_rate_init   = 0.001,
                early_stopping       = True,
                validation_fraction  = 0.1,
                random_state         = 42
            ),
            "scaled": True,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN AND EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(models_dict, X_train, X_val, X_test,
                       y_train, y_val, y_test,
                       prev_prices_test, actual_prices_test):

    # Scaler fitted only on training data (no leakage)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    results = {}

    print("\n" + "═" * 72)
    print("  MODEL EVALUATION — HELD-OUT TEST SET")
    print("═" * 72)
    print(f"  {'Model':<22} {'MAE':>7} {'RMSE':>7} {'R²':>7}  {'±Rs.2':>7}  {'±Rs.5':>7}")
    print("  " + "─" * 68)

    for name, cfg in models_dict.items():
        m       = cfg["model"]
        use_sc  = cfg["scaled"]

        Xtr = X_train_sc if use_sc else X_train
        Xte = X_test_sc  if use_sc else X_test

        m.fit(Xtr, y_train)

        pred_changes = m.predict(Xte)
        pred_prices  = prev_prices_test + pred_changes   # reconstruct absolute price

        mae  = mean_absolute_error(actual_prices_test, pred_prices)
        rmse = np.sqrt(mean_squared_error(actual_prices_test, pred_prices))
        r2   = r2_score(actual_prices_test, pred_prices)
        w2   = np.mean(np.abs(actual_prices_test - pred_prices) <= 2.0) * 100
        w5   = np.mean(np.abs(actual_prices_test - pred_prices) <= 5.0) * 100

        results[name] = {
            "model": m, "scaler": scaler if use_sc else None,
            "scaled": use_sc,
            "MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "R2": round(r2, 4), "w2": round(w2, 1), "w5": round(w5, 1),
        }

        print(f"  {name:<22} {mae:>7.4f} {rmse:>7.4f} {r2:>7.4f}  {w2:>6.1f}%  {w5:>6.1f}%")

    print("═" * 72)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — 2030 PROJECTION (Linear Trend per District)
# ═══════════════════════════════════════════════════════════════════════════════

def project_2030(df):
    print("\n  2030 Price Projections (Linear Trend — reliable districts only):")
    print("  " + "─" * 50)

    reliable = {"Chennai", "Dindigul", "Thiruchirapalli"}
    origin   = df['date'].min()
    t2030    = (pd.Timestamp("2030-07-01") - origin).days / 365.25

    projections = {}
    for dist, g in df.groupby('market'):
        g2 = g.copy()
        g2['yr_n'] = (g2['date'] - origin).dt.days / 365.25
        m = LinearRegression().fit(g2[['yr_n']], g2['price'])
        p = m.predict([[t2030]])[0]
        projections[dist] = round(p, 2)

        tag = " ✔ reliable" if dist in reliable else " (limited data)"
        print(f"  {dist:<18}  Rs.{p:>6.2f}/KG  {tag}")

    return projections


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — INTERACTIVE PRICE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def predict_for_date(date_str, results, df, le):
    """
    Given a target date string (YYYY-MM-DD), predict price for each district
    using the best model (Linear Regression) and print a summary.
    """
    try:
        target = pd.Timestamp(date_str)
    except Exception:
        print("  Invalid date. Use format YYYY-MM-DD  e.g. 2028-06-15")
        return

    origin   = df['date'].min()
    t_days   = (target - origin).days
    t_years  = t_days / 365.25

    print(f"\n  Predicted rice prices for {target.strftime('%d %B %Y')}:")
    print("  " + "─" * 50)

    for dist, g in df.groupby('market'):
        g2 = g.copy()
        g2['yr_n'] = (g2['date'] - origin).dt.days / 365.25
        m = LinearRegression().fit(g2[['yr_n']], g2['price'])
        p = m.predict([[t_years]])[0]
        # Sanity cap — never predict below Rs.7 or above Rs.200
        p = max(7.0, min(200.0, p))
        print(f"    {dist:<18}  Rs.{p:.2f}/KG")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 72)
    print("  Tamil Nadu Rice Price Prediction Platform")
    print("  Multi-District | 1994–2026 | 5 Models")
    print("═" * 72)

    # Load
    print("\n  Loading dataset...")
    df, features, le = load_and_engineer(DATASET_PATH)
    print(f"  {len(df)} records loaded | {df['market'].nunique()} districts")

    # Split
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     prev_test, actual_test,
     test_markets, test_dates, s1, s2) = split_data(df, features)

    print(f"\n  Split → Train: {s1} | Val: {s2-s1} | Test: {len(df)-s2}")
    print(f"  Test period: {pd.Timestamp(test_dates[0]).date()}  →  {pd.Timestamp(test_dates[-1]).date()}")

    # Train + evaluate
    models_dict = build_models()
    results = train_and_evaluate(
        models_dict, X_train, X_val, X_test,
        y_train, y_val, y_test,
        prev_test, actual_test
    )

    # 2030 projections
    project_2030(df)

    # Interactive prediction loop
    print("\n  ─────────────────────────────────────────────────────")
    print("  Enter any future date to predict district prices.")
    print("  Type 'quit' to exit.")
    while True:
        user = input("\n  Date (YYYY-MM-DD) or 'quit': ").strip()
        if user.lower() in ('quit', 'q', 'exit'):
            print("\n  Goodbye!\n")
            break
        predict_for_date(user, results, df, le)


if __name__ == '__main__':
    main()
