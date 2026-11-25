import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class MultiEvaluator:
    def __init__(self, datasets):
        self.datasets = datasets
        # Ensure we are looking in the root directory where your .txt files are
        self.base_path = r"C:\Users\selva\vio_project"
        if not os.path.exists("plots"):
            os.makedirs("plots")

    def find_gt_file(self, dataset_name):
        """Searches for ground truth data.csv in the raw data folders."""
        search_path = os.path.join(self.base_path, "data", "raw", dataset_name)
        for root, dirs, files in os.walk(search_path):
            if "data.csv" in files:
                # Prioritize folders containing 'groundtruth' or 'estimate'
                if "groundtruth" in root.lower() or "estimate" in root.lower():
                    return os.path.join(root, "data.csv")
        return None

    def evaluate(self):
        summary_results = []

        for ds in self.datasets:
            print(f"\n--- Processing: {ds} ---")
            
            # Match the exact filename format from your 'ls' output
            est_file = os.path.join(self.base_path, f"St_{ds}_Akram.txt")
            gt_file = self.find_gt_file(ds)

            if not os.path.exists(est_file):
                print(f"❌ Trajectory file not found: {est_file}")
                continue
            if not gt_file:
                print(f"❌ Ground Truth (data.csv) not found for {ds}")
                continue

            # 1. Load Estimated Trajectory
            est_data = pd.read_csv(est_file, sep=" ", header=None)
            est = est_data.values

            # 2. Load Ground Truth
            gt_df = pd.read_csv(gt_file)
            gt = gt_df.values
            if gt[0, 0] > 1e12: gt[:, 0] /= 1e9 # Convert ns to s

            # 3. Calculate ATE (RMSE)
            errors = []
            for row in est:
                # Temporal synchronization
                idx = np.argmin(np.abs(gt[:, 0] - row[0]))
                gt_pos = gt[idx, 1:4]
                est_pos = row[1:4]
                errors.append(np.linalg.norm(gt_pos - est_pos))
            
            rmse = np.sqrt(np.mean(np.square(errors)))
            summary_results.append({"Dataset": ds, "RMSE": rmse})
            print(f"✅ Success! RMSE: {rmse:.4f} m")

            # 4. Generate Plot for Week 12 Report
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(est[:, 1], est[:, 3], label='VIO Path', color='blue')
            plt.title(f"Top-Down (X-Z): {ds}")
            plt.xlabel("X (m)")
            plt.ylabel("Z (m)")
            plt.grid(True, linestyle=':')

            plt.subplot(1, 2, 2)
            plt.plot(est[:, 0] - est[0, 0], est[:, 2], label='Altitude', color='green')
            plt.title("Height Stability (Y)")
            plt.xlabel("Time (s)")
            plt.ylabel("Height (m)")
            plt.grid(True, linestyle=':')
            
            plt.tight_layout()
            plt.savefig(f"plots/Final_Result_{ds}.png")
            plt.close()

        # Final Summary Table
        print("\n" + "="*45)
        print(f"{'M1 PROJECT FINAL PERFORMANCE':^45}")
        print("="*45)
        print(f"{'Dataset':<28} | {'RMSE (m)':<10}")
        print("-" * 45)
        for res in summary_results:
            print(f"{res['Dataset']:<28} | {res['RMSE']:<10.4f}")
        print("="*45)

if __name__ == "__main__":
    target_datasets = [
        "dataset-room2_512_16",
        "dataset-corridor3_512_16",
        "dataset-outdoors5_512_16"
    ]
    evaluator = MultiEvaluator(target_datasets)
    evaluator.evaluate()