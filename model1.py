import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# =========================
# LOAD
# =========================
df = pd.read_csv(r"C:\Users\shris\OneDrive\Desktop\research\stroke_dataset_final.csv")
df = df.dropna(subset=["volume"])

# =========================
# TARGET
# =========================
y = np.log(df["volume"] + 1)

# =========================
# FEATURES (SPI BASE)
# =========================
features = ["dwi_mean", "adc_mean", "flair_mean"]
X = df[features]

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# SCALE
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# SPI (α, β, γ)
# =========================
lin = LinearRegression()
lin.fit(X_train_scaled, y_train)

alpha, beta, gamma = lin.coef_

print("\n=== SPI EQUATION ===")
print(f"SPI = {alpha:.4f}*DWI + {beta:.4f}*ADC + {gamma:.4f}*FLAIR")

# =========================
# CREATE SPI FEATURE
# =========================
spi_train = lin.predict(X_train_scaled)
spi_test = lin.predict(X_test_scaled)

# =========================
# 🔥 ADD IMPORTANT FEATURES BACK
# =========================
extra_features = ["dwi_std", "adc_std", "flair_std"]

X_extra = df[extra_features]

X_extra_train, X_extra_test = train_test_split(
    X_extra, test_size=0.2, random_state=42
)

# =========================
# FINAL FEATURE SET
# =========================
X_train_final = np.column_stack((spi_train, X_extra_train))
X_test_final = np.column_stack((spi_test, X_extra_test))

# =========================
# RANDOM FOREST
# =========================
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    random_state=42
)

rf.fit(X_train_final, y_train)
y_pred = rf.predict(X_test_final)

# =========================
# RESULTS
# =========================
print("\n--- FINAL MODEL ---")
print("R2:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# =========================
# GRAPH
# =========================
plt.figure()
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.title("Actual vs Predicted")
plt.show()