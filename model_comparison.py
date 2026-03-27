import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import warnings

# Hide warnings for a clean output
warnings.filterwarnings("ignore")

# 1. Load Data
file_path = 'TamilNadu_Rice_Refined_Unit_Standardized.xlsx'
df = pd.read_excel(file_path)

# 2. Preprocess: Create State-Wide Average Trend
state_avg = df.groupby('date')['price'].mean().reset_index()
state_avg['date_numeric'] = pd.to_datetime(state_avg['date']).map(pd.Timestamp.toordinal)

X = state_avg[['date_numeric']]
y = state_avg['price']

# 3. Initialize & Train all 5 Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR (Support Vector)": SVR(kernel='rbf', C=100),
    "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
}

mae_results = {}
for name, model in models.items():
    model.fit(X, y)
    mae_results[name] = mean_absolute_error(y, model.predict(X))

# 4. Show the Accuracy Table ONCE
print("\n" + "="*40)
print("   MODEL ACCURACY COMPARISON (MAE)")
print("="*40)
for name, mae in mae_results.items():
    print(f"{name:<22}: ±₹{mae:.2f}")
print("="*40)

# 5. The Prediction Loop
while True:
    print("\n" + "-"*60)
    user_input = input("Enter Future Date (YYYY-MM-DD) or type 'quit' to exit: ").strip().lower()
    
    if user_input == 'quit':
        print("Exiting predictor. Thank you!")
        break
    
    try:
        # Parse the date and convert to numeric
        target_date = pd.to_datetime(user_input)
        d_num = target_date.toordinal()
        X_test = pd.DataFrame([[d_num]], columns=['date_numeric'])

        print(f"\nCOMPARING 5 MODELS FOR: {user_input}")
        print(f"{'Model Name':<22} | {'Trend Price':<12} | {'Range (±MAE)'}")
        print("-"*60)

        for name, model in models.items():
            point_pred = model.predict(X_test)[0]
            mae = mae_results[name]
            print(f"{name:<22} | ₹{point_pred:>10.2f} | ₹{point_pred-mae:.2f} to ₹{point_pred+mae:.2f}")
        print("-" * 60)

    except Exception as e:
        print(f"Error: Invalid date format. Please use YYYY-MM-DD (e.g., 2030-01-01).")