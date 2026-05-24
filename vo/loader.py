import os
import cv2
import numpy as np
import pandas as pd


# =========================
# LOAD IMAGES
# =========================

def load_images(folder):

    files = sorted(os.listdir(folder))
    images = []

    print(f"[INFO] Loading {len(files)} images...")

    for i, f in enumerate(files):

        path = os.path.join(folder, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            images.append(img)

        if i % 500 == 0:
            print(f"[INFO] Loaded {i}/{len(files)}")

    print("[INFO] Image loading done.")
    return images


# =========================
# LOAD IMU
# =========================

def load_imu(csv_path):

    df = pd.read_csv(csv_path)

    print(f"[INFO] IMU loaded: {len(df)} samples")

    return df


# =========================
# LOAD GROUND TRUTH
# =========================

def load_groundtruth(csv_path):

    df = pd.read_csv(csv_path)

    cols = [c.lower() for c in df.columns]

    # Try common EuRoC / SLAM formats
    candidates = [
        ("p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]"),
        ("pos_x", "pos_y", "pos_z"),
        ("x", "y", "z"),
        ("px", "py", "pz"),
    ]

    for c in candidates:
        if all(any(cj.lower() == col for col in df.columns) for cj in c):
            print("[INFO] GT format detected:", c)
            return df[list(c)].values

    # 🔥 LAST RESORT: assume last 3 columns are position
    print("[WARNING] Unknown format → using last 3 columns")
    return df.iloc[:, -3:].values