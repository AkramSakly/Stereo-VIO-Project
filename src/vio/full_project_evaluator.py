import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

class MultiEvaluator:
    def __init__(self, datasets):
        self.datasets = datasets
        if not os.path.exists("plots"):
            os.makedirs("plots")

    def find_gt_file(self, dataset_name):
        """Searches for ground truth data.csv in any subdirectory."""
        base_dir = os.path.join("data", "raw", dataset_name)
        for root, dirs, files in os.walk(base_dir):
            if "data.csv" in files:
                # Prioritize groundtruth folders
                if "groundtruth" in root.lower() or "estimate0" in root.lower():
                    return os.path.join(root, "data.csv")
        return None

    def evaluate(self):
        summary_results = []

        for ds in self.datasets:
            print(f"\n--- Evaluating {ds} ---")
            est_file = f"St_{ds}_Akram.txt"
            gt_file = self.find_gt_file(ds)

            if not os.path.exists(est_file) or not gt_file:
                print(f"⚠️ Skipping {ds}: Missing files.")
                continue

            # 1. Load Data
            est = pd.read_csv(est_file, sep=" ", header=None).values
            gt = pd.read_csv(gt_file).values
            if gt[0, 0] > 1e12: gt[:, 0] /= 1e9 # Convert ns to s

            # 2. Calculate ATE (RMSE)
            errors = []
            for row in est:
                idx = np.argmin(np.abs(gt[:, 0] - row[0]))
                gt_pos = gt[idx, 1:4]
                est_pos = row[1:4]
                errors.append(np.linalg.norm(gt_pos - est_pos))
            
            rmse = np.sqrt(np.mean(np.square(errors)))
            summary_results.append({"Dataset": ds, "RMSE": rmse})
            print(f"✅ RMSE: {rmse:.4f} m")

            # 3. Plot Comparison
            data = pd.read_csv(est_file, sep=" ", header=None)
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(data[1], data[3], label='VIO Path')
            plt.title(f"{ds} (X-Z)")
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(data[0] - data[0][0], data[2], color='green')
            plt.title("Vertical Stability")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"plots/Report_{ds}.png")
            plt.close()

        # 4. Print Final Report Table
        print("\n" + "="*30)
        print("FINAL WEEK 11 PERFORMANCE TABLE")
        print("="*30)
        print(f"{'Dataset':<25} | {'RMSE (ATE)':<10}")
        print("-" * 40)
        for res in summary_results:
            print(f"{res['Dataset']:<25} | {res['RMSE']:<10.4f} m")
        print("="*30)

if __name__ == "__main__":
    my_datasets = [
        "dataset-room2_512_16",
        "dataset-corridor3_512_16",
        "dataset-outdoors5_512_16"
    ]
    MultiEvaluator(my_datasets).evaluate()