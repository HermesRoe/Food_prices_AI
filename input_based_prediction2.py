import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

def train_models(filepath):
    df = pd.read_excel(filepath)
    df['date'] = pd.to_datetime(df['date'])
    ts = df.groupby('date')['price'].mean().reset_index().sort_values('date').reset_index(drop=True)
    ts.columns = ['ds', 'y']
    ts['t'] = np.arange(len(ts))
    t = ts['t'].values
    y = ts['y'].values

    # Model 1: Linear
    m1 = LinearRegression()
    m1.fit(t.reshape(-1,1), y)

    # Model 2: Polynomial (deg 2)
    coeffs = np.polyfit(t, y, deg=2)
    poly = np.poly1d(coeffs)

    # Model 3: Exponential
    def exp_func(t, a, b, c):
        return a * np.exp(b * t) + c
    popt, _ = curve_fit(exp_func, t, y, p0=[7, 0.005, 0], maxfev=10000)

    # Model 4: Prophet-style (Polynomial + Fourier Seasonality)
    def fourier_feats(t_vals, months, period=12, n=3):
        feats = [t_vals, t_vals**2]
        for k in range(1, n+1):
            feats.append(np.sin(2*np.pi*k*months/period))
            feats.append(np.cos(2*np.pi*k*months/period))
        return np.column_stack(feats)

    months_all = ts['ds'].dt.month.values
    X4 = fourier_feats(t, months_all)
    m4 = Ridge(alpha=0.1)
    m4.fit(X4, y)

    print(f"✅ All 4 models trained on {len(ts)} records ({ts['ds'].min().date()} → {ts['ds'].max().date()})")
    return {
        'linear': m1,
        'polynomial': poly,
        'exponential': (exp_func, popt),
        'prophet': (m4, fourier_feats),
        'last_t': len(ts),
        'last_date': ts['ds'].max()
    }

# ── Predict for a given date ─────────────────────────────────────────────────
def predict(models, date_str):
    try:
        target = pd.Timestamp(date_str)
    except Exception:
        print("❌ Invalid date format. Use YYYY-MM-DD")
        return

    last_date = models['last_date']
    last_t    = models['last_t']

    # Calculate t index for target date
    months_ahead = (target.year - last_date.year)*12 + (target.month - last_date.month)
    kt = last_t + months_ahead

    if kt < 0:
        print("⚠️  Date is before training data start. Extrapolating backwards.")

    # Predictions
    lr_p   = models['linear'].predict([[kt]])[0]
    poly_p = models['polynomial'](kt)

    exp_func, popt = models['exponential']
    exp_p = exp_func(kt, *popt)

    m4, fourier_feats = models['prophet']
    Xf    = fourier_feats(np.array([kt]), np.array([target.month]))
    proph_p = m4.predict(Xf)[0]

    print(f"\n📅 Date       : {target.strftime('%d %B %Y')}")
    print(f"{'─'*40}")
    print(f"  Linear Regression  : ₹{lr_p:.2f} / KG")
    print(f"  Polynomial         : ₹{poly_p:.2f} / KG")
    print(f"  Exponential Growth : ₹{exp_p:.2f} / KG")
    print(f"  Prophet-style      : ₹{proph_p:.2f} / KG")
    print(f"{'─'*40}")
    # Best estimate: average of Poly + Prophet (most reliable for extrapolation)
    best_est = (poly_p + proph_p) / 2
    print(f"  ✅ Best Estimate   : ₹{best_est:.2f} / KG  (Poly + Prophet avg)")

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    DATASET_PATH = './TamilNadu_Rice_Refined_Unit_Standardized.xlsx'

    print("="*50)
    print("  Tamil Nadu Rice Price Predictor")
    print("="*50)

    models = train_models(DATASET_PATH)

    print("\nEnter a date to predict rice price (or 'quit' to exit)")
    while True:
        user_input = input("\nEnter date (YYYY-MM-DD) or 'quit': ").strip()
        if user_input.lower() == 'quit':
            print("Goodbye! 👋")
            break
        predict(models, user_input)
