import numpy as np
import pandas as pd
import nibabel as nib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# =========================================================
# 📂 LOAD DATASET (CSV)
# =========================================================
df = pd.read_csv(r"C:\Users\shris\OneDrive\Desktop\research\stroke_dataset_final.csv")

# =========================================================
# 🧹 CLEAN
# =========================================================
df["age"] = pd.to_numeric(df["age"], errors='coerce')
df = df.dropna(subset=["volume"])
df = df.drop(columns=["patient", "sex"], errors='ignore')
df = df.fillna(df.mean(numeric_only=True))

# =========================================================
# 🎯 TARGET
# =========================================================
y = np.log(df["volume"] + 1)

features = ["dwi_mean", "adc_mean", "flair_mean"]

# =========================================================
# 🔥 STEP 1: LEARN SPI WEIGHTS (SCALED)
# =========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

lin = LinearRegression()
lin.fit(X_scaled, y)

alpha, beta, gamma = lin.coef_

print("\nLearned Weights:")
print("DWI:", alpha)
print("ADC:", beta)
print("FLAIR:", gamma)

# =========================================================
# 🔥 STEP 2: CREATE SPI
# =========================================================
df["spi"] = (
    alpha * df["dwi_mean"] +
    beta * df["adc_mean"] +
    gamma * df["flair_mean"]
)

# =========================================================
# 🤖 STEP 3: TRAIN MODEL
# =========================================================
X = df[["spi"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- MODEL PERFORMANCE ---")
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# =========================================================
# 🔮 FUNCTION: USER INPUT (SCANS → OUTPUT)
# =========================================================
def predict_from_scans(dwi_path, adc_path, flair_path, mask_path):

    # -------- USER INPUT --------
    patient_id = input("Enter Patient ID: ")
    age = float(input("Enter Age: "))
    weight = float(input("Enter Weight (kg): "))


    # -------- LOAD SCANS --------
    dwi = nib.load(dwi_path).get_fdata()
    adc = nib.load(adc_path).get_fdata()
    flair = nib.load(flair_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    mask_bin = mask > 0

    # -------- FEATURES --------
    dwi_mean = np.mean(dwi[mask_bin])
    adc_mean = np.mean(adc[mask_bin])
    flair_mean = np.mean(flair[mask_bin]) 
    volume = np.sum(mask_bin)

    coords = np.argwhere(mask_bin)
    spread = np.prod(coords.max(axis=0) - coords.min(axis=0))

    # -------- SPI --------
    spi = alpha * dwi_mean + beta * adc_mean + gamma * flair_mean

    # -------- PREDICTION --------
    pred = model.predict(pd.DataFrame([[spi]], columns=["spi"]))[0]

    # -------- PRINT --------
    print("\n===== PATIENT INFO =====")
    print("Patient ID:", patient_id)
    print("Age:", age)

    return {
        "Patient ID": patient_id,
        "Age": age,
        "Weight": weight,
        "DWI Mean": dwi_mean,
        "ADC Mean": adc_mean,
        "FLAIR Mean": flair_mean,
        "Spread": spread,
        "SPI": spi,
        "Predicted Severity (log)": pred,
        "Estimated Lesion Volume": np.exp(pred)
    }

# =========================================================
# ✋ USER INPUT (YOUR WINDOWS PATHS)
# =========================================================
result = predict_from_scans(
    dwi_path = r"C:\Users\shris\OneDrive\Desktop\research\150 patients\Test\sub-strokecase0203\ses-0001\dwi\sub-strokecase0203_ses-0001_dwi.nii.gz",
    
    adc_path = r"C:\Users\shris\OneDrive\Desktop\research\150 patients\Test\sub-strokecase0203\ses-0001\dwi\sub-strokecase0203_ses-0001_adc.nii.gz",
    
    flair_path = r"C:\Users\shris\OneDrive\Desktop\research\150 patients\Test\sub-strokecase0203\ses-0001\anat\sub-strokecase0203_ses-0001_FLAIR.nii.gz",
    
    mask_path = r"C:\Users\shris\OneDrive\Desktop\research\150 patients\Test\derivatives testing\sub-strokecase0203\ses-0001\sub-strokecase0203_ses-0001_msk.nii.gz"
)

print("\nOUTPUT:\n", result)