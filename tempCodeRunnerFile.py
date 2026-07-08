import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay

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
# 🔴 SPI (CORE IDEA)
# =========================
features_spi = ["dwi_mean", "adc_mean", "flair_mean"]
X_spi_base = df[features_spi]

# -------- DEBUG --------
print("\n===== DEBUG INFO =====")
print("Shape:", X_spi_base.shape)
print("\nFirst rows:\n", X_spi_base.head())
print("\nVariance:\n", X_spi_base.var())

# -------- REMOVE LOW VARIANCE --------
X_spi_base = X_spi_base.loc[:, X_spi_base.var() > 1e-8]

print("\nRemaining features after variance filter:", X_spi_base.columns)

# -------- TRAIN LINEAR MODEL --------
lin = LinearRegression()
lin.fit(X_spi_base, y)

coefs = lin.coef_

print("\nRaw coefficients:", coefs)
print("Number of coefficients:", len(coefs))

# -------- SAFE ASSIGNMENT --------
if len(coefs) == 3:
    alpha, beta, gamma = coefs
else:
    print("⚠️ Warning: Expected 3 features but got", len(coefs))
    # fallback (avoid crash)
    alpha = coefs[0] if len(coefs) > 0 else 0
    beta  = coefs[1] if len(coefs) > 1 else 0
    gamma = coefs[2] if len(coefs) > 2 else 0

print("\n=== SPI WEIGHTS ===")
print("Alpha (DWI):", alpha)
print("Beta (ADC):", beta)
print("Gamma (FLAIR):", gamma)

# =========================
# CREATE SPI
# =========================
df["spi"] = (
    alpha * df["dwi_mean"] +
    beta * df["adc_mean"] +
    gamma * df["flair_mean"]
)

# =========================
# 🔵 SINGLE MODEL
# =========================
X_single = df[["dwi_mean"]]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_single, y, test_size=0.2, random_state=42
)

model_single = RandomForestRegressor(n_estimators=300, random_state=42)
model_single.fit(X_train_s, y_train_s)
y_pred_single = model_single.predict(X_test_s)

# =========================
# 🔴 SPI MODEL
# =========================
X_spi = df[["spi"]]

X_train_spi, X_test_spi, y_train_spi, y_test_spi = train_test_split(
    X_spi, y, test_size=0.2, random_state=42
)

model_spi = RandomForestRegressor(n_estimators=300, random_state=42)
model_spi.fit(X_train_spi, y_train_spi)
y_pred_spi = model_spi.predict(X_test_spi)

# =========================
# 🔥 MULTIMODAL MODEL
# =========================
features_multi = [
    "dwi_mean", "adc_mean", "flair_mean",
    "dwi_std", "adc_std", "flair_std",
    "dwi_contrast", "adc_contrast", "flair_contrast"
]

X_multi = df[features_multi]

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)

model_multi = RandomForestRegressor(
    n_estimators=500,
    max_depth=12,
    random_state=42
)

model_multi.fit(X_train_m, y_train_m)
y_pred_multi = model_multi.predict(X_test_m)

# =========================
# 📊 RESULTS
# =========================
print("\n--- MODEL COMPARISON ---")

print("\nSingle (DWI):")
print("R2:", r2_score(y_test_s, y_pred_single))
print("RMSE:", np.sqrt(mean_squared_error(y_test_s, y_pred_single)))

print("\nSPI Model:")
print("R2:", r2_score(y_test_spi, y_pred_spi))
print("RMSE:", np.sqrt(mean_squared_error(y_test_spi, y_pred_spi)))

print("\nMultimodal Model:")
print("R2:", r2_score(y_test_m, y_pred_multi))
print("RMSE:", np.sqrt(mean_squared_error(y_test_m, y_pred_multi)))

# =========================
# 📊 VISUALIZATION
# =========================

# MODEL COMPARISON
models = ["Single", "SPI", "Multimodal"]
scores = [
    r2_score(y_test_s, y_pred_single),
    r2_score(y_test_spi, y_pred_spi),
    r2_score(y_test_m, y_pred_multi)
]

plt.figure()
plt.bar(models, scores)
plt.ylabel("R2 Score")
plt.title("Model Comparison")
plt.show()

# ACTUAL VS PREDICTED (SPI)
plt.figure()
plt.scatter(y_test_spi, y_pred_spi)

plt.plot([y_test_spi.min(), y_test_spi.max()],
         [y_test_spi.min(), y_test_spi.max()])

plt.title("SPI Model: Actual vs Predicted")
plt.show()

# SPI vs VOLUME
plt.figure()
plt.scatter(df["spi"], y)
plt.title("SPI vs Volume")
plt.show()

# =========================
# 🔴 CONFUSION MATRIX
# =========================
threshold = np.percentile(y_test_spi, 60)

y_class = (y_test_spi > threshold).astype(int)
y_pred_class = (y_pred_spi > threshold).astype(int)

cm = confusion_matrix(y_class, y_pred_class)
ConfusionMatrixDisplay(cm).plot()

plt.title("Confusion Matrix (SPI Model)")
plt.show()

# =========================
# 🔮 PREDICTION FUNCTION
# =========================
def predict_from_scans(dwi_path, adc_path, flair_path, mask_path):

    dwi = nib.load(dwi_path).get_fdata()
    adc = nib.load(adc_path).get_fdata()
    flair = nib.load(flair_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()

    mask_bin = mask > 0

    dwi_mean = np.mean(dwi[mask_bin])
    adc_mean = np.mean(adc[mask_bin])
    flair_mean = np.mean(flair[mask_bin])

    spi = alpha * dwi_mean + beta * adc_mean + gamma * flair_mean

    pred = model_spi.predict([[spi]])[0]

    print("\nSPI:", spi)
    print("Predicted Log Volume:", pred)
    print("Estimated Volume:", np.exp(pred))

# =========================
# TEST
# =========================
predict_from_scans(
    dwi_path=r"C:\Users\shris\OneDrive\Desktop\research\150 patients\Test\sub-strokecase0203\ses-0001\dwi\sub-strokecase0203_ses-0001_dwi.nii.gz",
    adc_path=r"C:\Users\shris\OneDrive\Desktop\research\150 patients\Test\sub-strokecase0203\ses-0001\dwi\sub-strokecase0203_ses-0001_adc.nii.gz",
    flair_path=r"C:\Users\shris\OneDrive\Desktop\research\150 patients\Test\sub-strokecase0203\ses-0001\anat\sub-strokecase0203_ses-0001_FLAIR.nii.gz",
    mask_path=r"C:\Users\shris\OneDrive\Desktop\research\150 patients\Test\derivatives testing\sub-strokecase0203\ses-0001\sub-strokecase0203_ses-0001_msk.nii.gz"
)



# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score, mean_squared_error

# # =========================
# # LOAD DATA
# # =========================
# df = pd.read_csv(r"C:\Users\shris\OneDrive\Desktop\research\stroke_dataset_final.csv")

# # =========================
# # CLEAN
# # =========================
# df["age"] = pd.to_numeric(df["age"], errors='coerce')
# df = df.dropna(subset=["volume"])
# df = df.drop(columns=["patient", "sex"], errors='ignore')
# df = df.fillna(df.mean(numeric_only=True))

# # =========================
# # TARGET (LOG SCALE)
# # =========================
# y = np.log(df["volume"] + 1)

# # =========================
# # 🔴 SPI (ONLY 3 FEATURES)
# # =========================
# features = ["dwi_mean", "adc_mean", "flair_mean"]
# X = df[features]

# # ✅ NORMALIZE (THIS FIXES YOUR ISSUE)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # =========================
# # LEARN α, β, γ
# # =========================
# lin = LinearRegression()
# lin.fit(X_scaled, y)

# alpha, beta, gamma = lin.coef_

# print("\n=== SPI WEIGHTS ===")
# print("Alpha (DWI):", alpha)
# print("Beta (ADC):", beta)
# print("Gamma (FLAIR):", gamma)

# # =========================
# # CREATE SPI (IMPORTANT: USE SCALED VALUES)
# # =========================
# df["spi"] = lin.predict(X_scaled)

# # =========================
# # MODEL ON SPI
# # =========================
# X_spi = df[["spi"]]

# X_train, X_test, y_train, y_test = train_test_split(
#     X_spi, y, test_size=0.2, random_state=42
# )

# model = RandomForestRegressor(n_estimators=300, random_state=42)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("\n--- SPI MODEL PERFORMANCE ---")
# print("R2:", r2_score(y_test, y_pred))
# print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# # =========================
# # GRAPH (IMPORTANT FOR PRESENTATION)
# # =========================
# plt.figure()
# plt.scatter(y_test, y_pred)
# plt.plot([y_test.min(), y_test.max()],
#          [y_test.min(), y_test.max()])
# plt.title("Actual vs Predicted (SPI)")
# plt.show()

# # =========================
# # SPI vs VOLUME
# # =========================
# plt.figure()
# plt.scatter(df["spi"], y)
# plt.title("SPI vs Volume")
# plt.xlabel("SPI")
# plt.ylabel("Log Volume")
# plt.show()