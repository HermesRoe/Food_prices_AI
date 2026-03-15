import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_excel("./TamilNadu_Rice_Refined_Unit_Standardized.xlsx")

# 2. Filter and Preprocess for Chennai Market
df_chennai = df[df['market'] == 'Chennai'].copy()
df_chennai['date'] = pd.to_datetime(df_chennai['date'])
df_chennai = df_chennai.sort_values('date').reset_index(drop=True)

# 3. Feature Engineering
# Predict price CHANGE instead of absolute price (handles structural price shifts)
df_chennai['price_change'] = df_chennai['price'].diff(1)

df_chennai['year'] = df_chennai['date'].dt.year
df_chennai['month'] = df_chennai['date'].dt.month
df_chennai['day_of_week'] = df_chennai['date'].dt.dayofweek

# Lag features on price change (momentum signals)
for i in [1, 2, 3, 7, 14]:
    df_chennai[f'change_lag_{i}'] = df_chennai['price_change'].shift(i)

# Lag features on absolute price (current level context)
for i in [1, 2, 3, 7, 14, 30]:
    df_chennai[f'price_lag_{i}'] = df_chennai['price'].shift(i)

# Rolling statistics
df_chennai['rolling_mean_7'] = df_chennai['price'].rolling(window=7).mean()
df_chennai['rolling_mean_30'] = df_chennai['price'].rolling(window=30).mean()
df_chennai['rolling_std_7'] = df_chennai['price'].rolling(window=7).std()

df_chennai = df_chennai.dropna()

# Define features and target (price change, NOT absolute price)
features = ['year', 'month', 'day_of_week',
            'change_lag_1', 'change_lag_2', 'change_lag_3', 'change_lag_7', 'change_lag_14',
            'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_7', 'price_lag_14', 'price_lag_30',
            'rolling_mean_7', 'rolling_mean_30', 'rolling_std_7']
X = df_chennai[features]
y = df_chennai['price_change']

# 4. Chronological Train-Test Split (80/20)
split_index = int(len(df_chennai) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
dates_test = df_chennai['date'].iloc[split_index:]

# 5. Train XGBoost with Early Stopping
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    early_stopping_rounds=50
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# 6. Predict and Reconstruct Absolute Prices
predicted_changes = model.predict(X_test)
# Reconstruct: previous day's actual price + predicted change
prev_prices = df_chennai['price'].iloc[split_index - 1:-1].values
predicted_prices = prev_prices + predicted_changes
actual_prices = df_chennai['price'].iloc[split_index:].values

mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))

print("--- XGBoost Price-Change Model ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Best Iteration: {model.best_iteration}")

# 7. Visualization
plt.figure(figsize=(14, 6))
plt.plot(dates_test.values, actual_prices,
         label='Actual Price', color='#2563eb', linewidth=2)
plt.plot(dates_test.values, predicted_prices, label='Predicted Price',
         color='#dc2626', linestyle='--', linewidth=2, alpha=0.85)
plt.xlabel('Date')
plt.ylabel('Price per KG (INR)')
plt.title('Actual vs Predicted Rice Prices in Chennai')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
