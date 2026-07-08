import os
import nibabel as nib
import numpy as np
import json
import pandas as pd
from scipy.ndimage import zoom
import glob

# -------- BASE --------
base = r"C:\Users\shris\Downloads\150 patients\150 patients\Train"
derivatives = os.path.join(base, "derivatives training")

# -------- RESIZE FUNCTION --------
def resize_to_match(source, target_shape):
    factors = [t / s for s, t in zip(source.shape, target_shape)]
    return zoom(source, factors, order=0)

# -------- FILE FINDER --------
def get_files(folder):
    data = {"dwi": None, "adc": None, "flair": None}

    if not os.path.exists(folder):
        return data

    for f in os.listdir(folder):
        f_lower = f.lower()
        if "dwi" in f_lower:
            data["dwi"] = f
        elif "adc" in f_lower:
            data["adc"] = f
        elif "flair" in f_lower:
            data["flair"] = f

    return data

# -------- PATIENT LIST --------
all_patients = sorted([p for p in os.listdir(base) if p.startswith("sub-")])

rows = []

# -------- LOOP --------
for patient in all_patients:
    try:
        ses = "ses-0001"

        anat_path = os.path.join(base, patient, ses, "anat")
        dwi_path = os.path.join(base, patient, ses, "dwi")
        mask_path = os.path.join(derivatives, patient, ses, f"{patient}_{ses}_msk.nii.gz")

        if not os.path.exists(mask_path):
            print("⚠️ No mask:", patient)
            continue

        dwi_files = get_files(dwi_path)
        anat_files = get_files(anat_path)

        if not dwi_files["dwi"] or not dwi_files["adc"] or not anat_files["flair"]:
            print("⚠️ Missing MRI files:", patient)
            continue

        # -------- LOAD --------
        dwi = nib.load(os.path.join(dwi_path, dwi_files["dwi"])).get_fdata()
        adc = nib.load(os.path.join(dwi_path, dwi_files["adc"])).get_fdata()
        flair = nib.load(os.path.join(anat_path, anat_files["flair"])).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # -------- RESIZE --------
        if adc.shape != dwi.shape:
            adc = resize_to_match(adc, dwi.shape)

        if flair.shape != dwi.shape:
            flair = resize_to_match(flair, dwi.shape)

        if mask.shape != dwi.shape:
            mask = resize_to_match(mask, dwi.shape)

        mask_bin = (mask > 0)
        volume = np.sum(mask_bin)

        if volume == 0:
            continue

        # -------- FEATURES --------
        dwi_mean = np.mean(dwi[mask_bin])
        adc_mean = np.mean(adc[mask_bin])
        flair_mean = np.mean(flair[mask_bin])

        dwi_std = np.std(dwi[mask_bin])
        adc_std = np.std(adc[mask_bin])
        flair_std = np.std(flair[mask_bin])

        dwi_adc_ratio = dwi_mean / (adc_mean + 1e-6)
        flair_adc_ratio = flair_mean / (adc_mean + 1e-6)

        coords = np.argwhere(mask_bin)
        x_min, y_min, z_min = coords.min(axis=0)
        x_max, y_max, z_max = coords.max(axis=0)

        x_size = x_max - x_min
        y_size = y_max - y_min
        z_size = z_max - z_min

        spread = x_size * y_size * z_size

        dwi_global = np.mean(dwi)
        adc_global = np.mean(adc)
        flair_global = np.mean(flair)

        dwi_contrast = dwi_mean - dwi_global
        adc_contrast = adc_mean - adc_global
        flair_contrast = flair_mean - flair_global

        spi = (dwi_mean - adc_mean + flair_mean) * np.log(volume + 1)

        # -------- JSON (CORRECT POSITION) --------
        age, sex, weight = None, None, None

        json_files = glob.glob(
            os.path.join(base, patient, ses, "**", "*.json"),
            recursive=True
        )

        if json_files:
            try:
                with open(json_files[0], 'r') as f:
                    meta = json.load(f)

                    raw_age = meta.get("PatientAge", None)
                    if raw_age:
                        age = int(str(raw_age).replace("Y", "").strip())

                    sex = meta.get("PatientSex", None)
                    weight = meta.get("PatientWeight", None)

            except Exception as e:
                print("⚠️ JSON error:", patient, e)

        # -------- STORE --------
        rows.append({
            "patient": patient,
            "age": age,
            "sex": sex,
            "weight": weight,
            "volume": volume,
            "dwi_mean": dwi_mean,
            "adc_mean": adc_mean,
            "flair_mean": flair_mean,
            "dwi_std": dwi_std,
            "adc_std": adc_std,
            "flair_std": flair_std,
            "dwi_adc_ratio": dwi_adc_ratio,
            "flair_adc_ratio": flair_adc_ratio,
            "x_size": x_size,
            "y_size": y_size,
            "z_size": z_size,
            "spread": spread,
            "dwi_contrast": dwi_contrast,
            "adc_contrast": adc_contrast,
            "flair_contrast": flair_contrast,
            "spi": spi
        })

        print("✅ Processed:", patient)

    except Exception as e:
        print("❌ Skipped:", patient, e)

# -------- SAVE --------
df = pd.DataFrame(rows)

df["sex"] = df["sex"].map({"M": 1, "F": 0})

df.to_csv("stroke_dataset_final.csv", index=False)

print("\n🎯 CSV Created!")
print("Total samples:", len(df))