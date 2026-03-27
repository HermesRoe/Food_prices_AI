import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import warnings

# Hide unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Load Data
file_path = 'TamilNadu_Rice_Refined_Unit_Standardized.xlsx'
df = pd.read_excel(file_path)

# 2. Group by date to get the State Average
state_avg = df.groupby('date')['price'].mean().reset_index()
state_avg['date_numeric'] = pd.to_datetime(state_avg['date']).map(pd.Timestamp.toordinal)

# 3. Train Model
X = state_avg[['date_numeric']]
y = state_avg['price']
model = LinearRegression().fit(X, y)

# 4. Calculate MAE (This is our 'Range' factor)
predictions_history = model.predict(X)
mae = mean_absolute_error(y, predictions_history)

print(f"--- LINEAR REGRESSION: STATE-WIDE TREND ---")
print(f"Model Accuracy (MAE): ₹{mae:.2f}")

# 5. Interactive Forecast with Range
user_date = input("\nEnter future date for Range Forecast (YYYY-MM-DD): ")
d_num = pd.to_datetime(user_date).toordinal()

# Predict the 'Point' price
point_price = model.predict(pd.DataFrame([[d_num]], columns=['date_numeric']))[0]

# Calculate the Range
lower_bound = point_price - mae
upper_bound = point_price + mae

print(f"\nResults for {user_date}:")
print(f"Predicted Trend Price: ₹{point_price:.2f}")
print(f"Expected Price Range:  ₹{lower_bound:.2f} to ₹{upper_bound:.2f}")
print(f"(Based on a historical error margin of ±₹{mae:.2f})")