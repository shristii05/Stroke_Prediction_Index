import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# -------- LOAD --------
df = pd.read_csv("stroke_dataset_final.csv")

# -------- CLEAN --------
df["age"] = pd.to_numeric(df["age"], errors='coerce')
df = df.dropna(subset=["volume"])
df = df.drop(columns=["patient", "sex"], errors='ignore')
df = df.fillna(df.mean(numeric_only=True))

# -------- TARGET --------
y = np.log(df["volume"] + 1)

# =========================================================
# 🔴 INDIVIDUAL MODELS
# =========================================================

features = ["dwi_mean", "adc_mean", "flair_mean"]

results = {}

for feature in features:
    print(f"\n--- MODEL USING {feature.upper()} ONLY ---")
    
    X = df[[feature]]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=200)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("R2 Score:", r2)
    print("RMSE:", rmse)
    
    results[feature] = r2

# =========================================================
# 📊 SUMMARY
# =========================================================
print("\n========== INDIVIDUAL MODEL COMPARISON ==========")
for k, v in results.items():
    print(f"{k}: R2 = {v:.4f}")