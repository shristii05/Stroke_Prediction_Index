import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error,
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, classification_report
)
from sklearn.preprocessing import StandardScaler

# =========================
# 📂 LOAD DATA
# =========================
df = pd.read_csv("stroke_dataset_final.csv")

# =========================
# 🧹 CLEAN
# =========================
df["age"] = pd.to_numeric(df["age"], errors='coerce')
df = df.dropna(subset=["volume"])
df = df.drop(columns=["patient", "sex"], errors='ignore')
df = df.fillna(df.mean(numeric_only=True))

# =========================
# 🔥 FEATURE ENGINEERING
# =========================
df["heterogeneity"] = df["dwi_std"] / (df["dwi_mean"] + 1e-6)
df["intensity_ratio"] = df["dwi_mean"] / (df["flair_adc_ratio"] + 1e-6)

# Normalize spread
df["spread_norm"] = df["spread"] / (df["spread"].mean() + 1e-6)

# =========================
# 🎯 TARGET
# =========================
y = np.log1p(df["volume"])

# =========================
# ⚖️ FEATURES
# =========================
core_features = [
    "spread_norm",
    "flair_adc_ratio",
    "dwi_mean",
    "dwi_std",
    "heterogeneity",
    "intensity_ratio"
]

X = df[core_features]

# =========================
# ⚖️ FORCE BALANCING (KEY STEP)
# =========================
feature_std = X.std()

# Inverse weighting → reduces dominant features
weights = 1 / (feature_std + 1e-6)

# Normalize weights
weights = weights / weights.mean()

# Apply weights
X_balanced = X * weights

# =========================
# SCALE
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# ⚖️ REGRESSION (BALANCED)
# =========================
model = Ridge(alpha=2.0)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n--- FINAL BALANCED MODEL ---")
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# =========================
# FEATURE IMPORTANCE
# =========================
importance_df = pd.DataFrame({
    "feature": core_features,
    "importance": np.abs(model.coef_)
}).sort_values("importance", ascending=False)

print("\n=== FEATURE IMPORTANCE (BALANCED) ===")
print(importance_df.to_string())

# =========================
# 🔴 SPI
# =========================
df["spi"] = model.predict(X_scaled)

# =========================
# 📊 VISUALIZATIONS
# =========================

# 1️⃣ Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Log Volume")
plt.ylabel("Predicted Log Volume")
plt.title("Balanced Model: Actual vs Predicted")
plt.show()

# 2️⃣ Feature Importance
plt.figure()
plt.barh(core_features, np.abs(model.coef_))
plt.xlabel("Importance")
plt.title("Balanced Feature Importance")
plt.show()

# 3️⃣ SPI vs Volume
plt.figure()
plt.scatter(df["spi"], y, alpha=0.6)

z = np.polyfit(df["spi"], y, 1)
p = np.poly1d(z)
plt.plot(df["spi"], p(df["spi"]), linestyle="--")

plt.xlabel("SPI")
plt.ylabel("Log Volume")
plt.title("Balanced SPI vs Volume")
plt.show()

# 4️⃣ Residual Plot
residuals = y_test - y_pred
plt.figure()
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# 5️⃣ Volume Distribution
plt.figure()
plt.hist(df["volume"], bins=30)
plt.title("Volume Distribution")
plt.xlabel("Volume")
plt.ylabel("Frequency")
plt.show()

# =========================
# 🔥 CLASSIFICATION
# =========================
low  = np.percentile(y, 30)
high = np.percentile(y, 70)

def classify(val):
    if val <= low:
        return 0
    elif val <= high:
        return 1
    else:
        return 2

y_class = np.array([classify(v) for v in y])

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_scaled, y_class, test_size=0.2, random_state=42
)

clf = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

clf.fit(X_train_c, y_train_c)
y_pred_class = clf.predict(X_test_c)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test_c, y_pred_class)

disp = ConfusionMatrixDisplay(cm, display_labels=["Mild", "Moderate", "Severe"])
disp.plot(cmap="Blues")
plt.title("Balanced Confusion Matrix")
plt.show()

print("\nConfusion Matrix:\n", cm)

# =========================
# METRICS
# =========================
accuracy = accuracy_score(y_test_c, y_pred_class)
precision = precision_score(y_test_c, y_pred_class, average='weighted')

print("\n=== FINAL BALANCED METRICS ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(
    y_test_c,
    y_pred_class,
    target_names=["Mild", "Moderate", "Severe"]
))