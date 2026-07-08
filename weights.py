import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
# 🧠 STEP 1: LEARN WEIGHTS
# =========================================================
features = ["dwi_mean", "adc_mean", "flair_mean"]

lin = LinearRegression()
lin.fit(df[features], y)

alpha, beta, gamma = lin.coef_

print("\nFinal Weights:")
print("Alpha:", alpha)
print("Beta:", beta)
print("Gamma:", gamma)

# =========================================================
# 🔥 STEP 2: CREATE BETTER SPI
# =========================================================
df["spi_learned"] = (
    alpha * df["dwi_mean"] +
    beta * df["adc_mean"] +
    gamma * df["flair_mean"]
) * np.log(df["spread"] + 1)   # ✅ use spread (NOT volume)

# =========================================================
# 🤖 STEP 3: TRAIN MODEL
# =========================================================
X = df[["spi_learned"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -------- RESULTS --------
print("\n--- FINAL SPI MODEL ---")
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))