import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom

# -------- BASE CONFIG --------
base = r"C:\Users\shris\Downloads\ISLES-2022 (1)\ISLES-2022"
derivatives = os.path.join(base, "derivatives")

patient = "sub-strokecase0007"
ses = "ses-0001"

anat_path = os.path.join(base, patient, ses, "anat")
dwi_path = os.path.join(base, patient, ses, "dwi")
mask_path = os.path.join(derivatives, patient, ses, f"{patient}_{ses}_msk.nii.gz")

# -------- FILE FINDER --------
def get_files(folder):
    files = os.listdir(folder)
    data = {}
    for f in files:
        f_lower = f.lower()
        if "dwi" in f_lower:
            data["dwi"] = f
        elif "adc" in f_lower:
            data["adc"] = f
        elif "flair" in f_lower:
            data["flair"] = f
    return data

dwi_files = get_files(dwi_path)
anat_files = get_files(anat_path)

# -------- LOAD --------
dwi = nib.load(os.path.join(dwi_path, dwi_files["dwi"])).get_fdata()
adc = nib.load(os.path.join(dwi_path, dwi_files["adc"])).get_fdata()
flair = nib.load(os.path.join(anat_path, anat_files["flair"])).get_fdata()
mask = nib.load(mask_path).get_fdata()

# -------- 1. MASK PARAMETERS --------
print("\n🔹 MASK ANALYSIS")

unique_vals = np.unique(mask)
print("Unique values:", unique_vals)

# Convert to binary (important)
mask_bin = (mask > 0).astype(np.uint8)

lesion_voxels = np.sum(mask_bin)
print("Lesion voxel count:", lesion_voxels)

# Approx volume (voxel count based)
print("Lesion volume (voxels):", lesion_voxels)

# Bounding box
coords = np.argwhere(mask_bin)
if coords.size > 0:
    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)
    
    print("Bounding box:")
    print(f"x: {x_min} → {x_max}")
    print(f"y: {y_min} → {y_max}")
    print(f"z: {z_min} → {z_max}")
else:
    print("No lesion found")

# -------- 2. IMAGE PARAMETERS --------
def get_stats(name, img):
    print(f"\n🔹 {name} STATS")
    print("Mean:", np.mean(img))
    print("Std:", np.std(img))
    print("Min:", np.min(img))
    print("Max:", np.max(img))

get_stats("DWI", dwi)
get_stats("ADC", adc)
get_stats("FLAIR", flair)

import os
import nibabel as nib
import numpy as np
import json
import pandas as pd

# -------- BASE --------
base = r"C:\Users\shris\Downloads\ISLES-2022 (1)\ISLES-2022"
derivatives = os.path.join(base, "derivatives")

# -------- FILE FINDER --------
def get_files(folder):
    files = os.listdir(folder)
    data = {}
    for f in files:
        f_lower = f.lower()
        if "dwi" in f_lower:
            data["dwi"] = f
        elif "adc" in f_lower:
            data["adc"] = f
        elif "flair" in f_lower:
            data["flair"] = f
    return data

# -------- PATIENTS --------
all_patients = [p for p in os.listdir(base) if p.startswith("sub-")]
all_patients.sort()

train_patients = all_patients[:150]

rows = []

# -------- LOOP --------
for patient in train_patients:
    try:
        ses = "ses-0001"

        anat_path = os.path.join(base, patient, ses, "anat")
        dwi_path = os.path.join(base, patient, ses, "dwi")
        mask_path = os.path.join(derivatives, patient, ses, f"{patient}_{ses}_msk.nii.gz")

        # Load files
        dwi_files = get_files(dwi_path)
        anat_files = get_files(anat_path)

        dwi = nib.load(os.path.join(dwi_path, dwi_files["dwi"])).get_fdata()
        adc = nib.load(os.path.join(dwi_path, dwi_files["adc"])).get_fdata()
        flair = nib.load(os.path.join(anat_path, anat_files["flair"])).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        mask_bin = (mask > 0)

        volume = np.sum(mask_bin)

        # Skip empty masks
        if volume == 0:
            continue

        # -------- LESION FEATURES --------
        dwi_l = dwi[mask_bin]
        adc_l = adc[mask_bin]
        flair_l = flair[mask_bin]

        dwi_mean = np.mean(dwi_l)
        adc_mean = np.mean(adc_l)
        flair_mean = np.mean(flair_l)

        dwi_std = np.std(dwi_l)
        adc_std = np.std(adc_l)
        flair_std = np.std(flair_l)

        # -------- RATIOS --------
        dwi_adc_ratio = dwi_mean / (adc_mean + 1e-6)
        flair_adc_ratio = flair_mean / (adc_mean + 1e-6)

        # -------- SPATIAL FEATURES --------
        coords = np.argwhere(mask_bin)
        x_min, y_min, z_min = coords.min(axis=0)
        x_max, y_max, z_max = coords.max(axis=0)

        x_size = x_max - x_min
        y_size = y_max - y_min
        z_size = z_max - z_min

        spread = x_size * y_size * z_size

        # -------- WHOLE BRAIN STATS --------
        dwi_global_mean = np.mean(dwi)
        adc_global_mean = np.mean(adc)
        flair_global_mean = np.mean(flair)

        # -------- CONTRAST FEATURES --------
        dwi_contrast = dwi_mean - dwi_global_mean
        adc_contrast = adc_mean - adc_global_mean
        flair_contrast = flair_mean - flair_global_mean

        # -------- SPI --------
        spi = (dwi_mean - adc_mean + flair_mean) * np.log(volume + 1)

        # -------- JSON --------
        json_path = os.path.join(base, patient, ses, f"{patient}_{ses}.json")

        age, sex = None, None
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                meta = json.load(f)
                age = meta.get("PatientAge", None)
                sex = meta.get("PatientSex", None)

        # -------- STORE --------
        rows.append({
            "patient": patient,

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

            "spi": spi,

            "age": age,
            "sex": sex
        })

        print("Processed:", patient)

    except Exception as e:
        print("❌ Skipping:", patient, e)

# -------- SAVE --------
df = pd.DataFrame(rows)
df.to_csv("stroke_dataset_upgraded.csv", index=False)

print("✅ Saved: stroke_dataset_upgraded.csv")
