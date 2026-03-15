import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
df = pd.read_excel("./TamilNadu_Rice_Refined_Unit_Standardized.xlsx")

# 2. Filter Chennai
df_chennai = df[df['market'] == 'Chennai'].copy()
df_chennai['date'] = pd.to_datetime(df_chennai['date'])
df_chennai = df_chennai.sort_values('date').reset_index(drop=True)

# 3. Feature Engineering
df_chennai['price_change'] = df_chennai['price'].diff(1)
df_chennai['year']         = df_chennai['date'].dt.year
df_chennai['month']        = df_chennai['date'].dt.month
df_chennai['day_of_week']  = df_chennai['date'].dt.dayofweek

for i in [1, 2, 3, 7, 14]:
    df_chennai[f'change_lag_{i}'] = df_chennai['price_change'].shift(i)

for i in [1, 2, 3, 7, 14, 30]:
    df_chennai[f'price_lag_{i}'] = df_chennai['price'].shift(i)

df_chennai['rolling_mean_7']  = df_chennai['price'].rolling(7).mean()
df_chennai['rolling_mean_30'] = df_chennai['price'].rolling(30).mean()
df_chennai['rolling_std_7']   = df_chennai['price'].rolling(7).std()
df_chennai = df_chennai.dropna().reset_index(drop=True)

features = [
    'year', 'month', 'day_of_week',
    'change_lag_1', 'change_lag_2', 'change_lag_3', 'change_lag_7', 'change_lag_14',
    'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_7', 'price_lag_14', 'price_lag_30',
    'rolling_mean_7', 'rolling_mean_30', 'rolling_std_7'
]

X = df_chennai[features]
y = df_chennai['price_change']

# 4. Train Model on ALL data (no split — use everything for best future prediction)
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)
model.fit(X, y, verbose=False)
print("Model trained. Ready for predictions.\n")

# 5. Drift correction values from last 90 days
recent           = df_chennai.tail(90)
avg_daily_change = recent['price_change'].mean()
clip_limit       = recent['price_change'].std() * 1.5

# 6. Predict price for any given date
def predict_price(target_date: str) -> float:
    target_dt       = pd.Timestamp(target_date)
    last_known_date = df_chennai['date'].max()

    # If date already exists in data, return actual
    if target_dt <= last_known_date:
        row = df_chennai[df_chennai['date'] == target_dt]
        if not row.empty:
            price = row['price'].values[0]
            print(f"  Date   : {target_date}  (historical record)")
            print(f"  Price  : Rs.{price:.2f} per KG")
            return price

    # Recursive forecast with drift correction
    history      = df_chennai[['date', 'price']].copy()
    current_date = last_known_date + pd.Timedelta(days=1)

    while current_date <= target_dt:
        prices  = history['price'].values
        changes = np.diff(prices)

        X_input = pd.DataFrame([{
            'year':          current_date.year,
            'month':         current_date.month,
            'day_of_week':   current_date.dayofweek,
            'change_lag_1':  changes[-1],
            'change_lag_2':  changes[-2],
            'change_lag_3':  changes[-3],
            'change_lag_7':  changes[-7],
            'change_lag_14': changes[-14],
            'price_lag_1':   prices[-1],
            'price_lag_2':   prices[-2],
            'price_lag_3':   prices[-3],
            'price_lag_7':   prices[-7],
            'price_lag_14':  prices[-14],
            'price_lag_30':  prices[-30],
            'rolling_mean_7':  prices[-7:].mean(),
            'rolling_mean_30': prices[-30:].mean(),
            'rolling_std_7':   prices[-7:].std(),
        }])[features]

        raw_change = model.predict(X_input)[0]
        blended    = (0.6 * raw_change) + (0.4 * avg_daily_change)
        clipped    = np.clip(blended, -clip_limit, clip_limit)
        pred_price = float(prices[-1] + clipped)

        history      = pd.concat([history, pd.DataFrame([{'date': current_date, 'price': pred_price}])], ignore_index=True)
        current_date += pd.Timedelta(days=1)

    final_price = history['price'].values[-1]
    print(f"  Date   : {target_date}")
    print(f"  Price  : Rs.{final_price:.2f} per KG")
    return final_price


# 7. Input loop
print("==============================")
print("  Rice Price Predictor")
print("  Market: Chennai")
print("==============================")

while True:
    user_input = input("\nEnter date (YYYY-MM-DD) or 'quit': ").strip()
    if user_input.lower() in ('quit', 'q', 'exit'):
        print("Goodbye!")
        break
    try:
        predict_price(user_input)
    except Exception as e:
        print(f"  Error: {e}")
        print("  Use format: YYYY-MM-DD  e.g. 2026-08-15")