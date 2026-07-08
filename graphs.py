import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("stroke_dataset_final.csv")

# CLEAN
df["age"] = pd.to_numeric(df["age"], errors='coerce')
df = df.dropna(subset=["volume"])
df = df.drop(columns=["patient", "sex"], errors='ignore')
df = df.fillna(df.mean(numeric_only=True))

# TARGET
y = np.log(df["volume"] + 1)

# =========================
# FEATURE SETS
# =========================
features = ["dwi_mean", "adc_mean", "flair_mean"]

# =========================
# TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df[features], y, test_size=0.2, random_state=42
)

# =========================
# 🔵 SINGLE MODEL (DWI)
# =========================
model_single = RandomForestRegressor(n_estimators=200)
model_single.fit(X_train[["dwi_mean"]], y_train)
y_pred_single = model_single.predict(X_test[["dwi_mean"]])

r2_single = r2_score(y_test, y_pred_single)

# =========================
# 🔵 MULTIMODAL MODEL
# =========================
model_multi = RandomForestRegressor(n_estimators=200)
model_multi.fit(X_train, y_train)
y_pred_multi = model_multi.predict(X_test)

r2_multi = r2_score(y_test, y_pred_multi)

# =========================
# 🔴 SPI MODEL
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

lin = LinearRegression()
lin.fit(X_scaled, y)

# SPI creation (correct way)
spi = lin.predict(X_scaled)
df["spi"] = spi

# train-test for SPI
X_spi = df[["spi"]]
X_train_spi, X_test_spi, y_train_spi, y_test_spi = train_test_split(
    X_spi, y, test_size=0.2, random_state=42
)

model_spi = RandomForestRegressor(n_estimators=200)
model_spi.fit(X_train_spi, y_train_spi)
y_pred_spi = model_spi.predict(X_test_spi)

r2_spi = r2_score(y_test_spi, y_pred_spi)

# =========================
# 📊 GRAPH 1: MODEL COMPARISON
# =========================
plt.figure()
models = ["Single (DWI)", "Multimodal", "SPI"]
scores = [r2_single, r2_multi, r2_spi]

plt.bar(models, scores)
plt.xlabel("Model")
plt.ylabel("R² Score")
plt.title("Model Comparison")
plt.show()

# =========================
# 📈 GRAPH 2: ACTUAL vs PREDICTED (SPI)
# =========================
plt.figure()
plt.scatter(y_test_spi, y_pred_spi)

plt.plot([y_test_spi.min(), y_test_spi.max()],
         [y_test_spi.min(), y_test_spi.max()])

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (SPI Model)")
plt.show()

# =========================
# 📉 GRAPH 3: RESIDUAL PLOT
# =========================
residuals = y_test_spi - y_pred_spi

plt.figure()
plt.scatter(y_pred_spi, residuals)
plt.axhline(y=0)

plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# =========================
# 📊 GRAPH 4: SPI vs VOLUME
# =========================
plt.figure()
plt.scatter(df["spi"], y)

plt.xlabel("SPI")
plt.ylabel("Log Volume")
plt.title("SPI vs Volume")
plt.show()

# =========================
# 📊 GRAPH 5: VOLUME DISTRIBUTION
# =========================
plt.figure()
plt.hist(df["volume"], bins=30)

plt.xlabel("Volume")
plt.title("Volume Distribution")
plt.show()

# =========================
# 📊 GRAPH 6: FEATURE IMPORTANCE (MULTIMODAL)
# =========================
importances = model_multi.feature_importances_

plt.figure()
plt.bar(features, importances)

plt.title("Feature Importance (Multimodal)")
plt.show()

# =========================
# 🔴 OPTIONAL: CONFUSION MATRIX
# =========================
y_class = (y_test_spi > y_test_spi.median()).astype(int)
y_pred_class = (y_pred_spi > y_test_spi.median()).astype(int)

cm = confusion_matrix(y_class, y_pred_class)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix (Stroke Severity)")
plt.show()