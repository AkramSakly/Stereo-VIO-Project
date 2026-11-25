import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_ate(gt_file, est_file):
    # Load data (assuming TUM format: timestamp x y z ...)
    gt = pd.read_csv(gt_file, sep=" ", header=None).values
    est = pd.read_csv(est_file, sep=" ", header=None).values

    # Sync timestamps (find closest matches)
    errors = []
    for row in est:
        # Find index of closest timestamp in ground truth
        idx = np.argmin(np.abs(gt[:,0] - row[0]))
        gt_pos = gt[idx, 1:4]
        est_pos = row[1:4]
        
        # Euclidean distance error
        errors.append(np.linalg.norm(gt_pos - est_pos))
    
    rmse = np.sqrt(np.mean(np.square(errors)))
    return rmse, errors

# Example usage for your Outdoors5 run
# rmse, err_list = calculate_ate("data/raw/dataset-outdoors5_512_16/mav0/state_groundtruth_estimate0/data.csv", "St_dataset-outdoors5_512_16_Akram.txt")
# print(f"Outdoors5 RMSE: {rmse:.4f} meters")