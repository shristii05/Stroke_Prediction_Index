import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(r"C:\Users\shris\OneDrive\Desktop\research\stroke_dataset_final.csv")

# =========================
# CLEAN
# =========================
df["age"] = pd.to_numeric(df["age"], errors='coerce')
df = df.dropna(subset=["volume"])
df = df.drop(columns=["patient", "sex"], errors='ignore')
df = df.fillna(df.mean(numeric_only=True))

# =========================
# TARGET
# =========================
y = np.log(df["volume"] + 1)

# =========================
# 🔴 INDIVIDUAL MODELS
# =========================
features = ["dwi_mean", "adc_mean", "flair_mean"]

results = {}

for feature in features:
    print(f"\n--- MODEL USING {feature.upper()} ---")

    X = df[[feature]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("R2:", r2)
    print("RMSE:", rmse)

    results[feature] = r2

# =========================
# 📊 COMPARISON GRAPH
# =========================
plt.figure()
plt.bar(results.keys(), results.values())
plt.ylabel("R2 Score")
plt.title("Single Modality Comparison")
plt.show()