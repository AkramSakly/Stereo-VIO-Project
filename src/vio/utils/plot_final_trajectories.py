import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class VIOPlotter:
    def __init__(self, base_path=r"C:\Users\selva\vio_project"):
        self.base_path = base_path
        # Separate folder to keep your presentation media organized
        self.output_dir = os.path.join(self.base_path, "exported_media", "final_plots")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 Created plots folder at: {self.output_dir}")

    def load_trajectory(self, filename):
        file_path = os.path.join(self.base_path, filename)
        if not os.path.exists(file_path):
            print(f"⚠️ Trajectory file missing: {filename}")
            return None
        
        # TUM-VI output traces format: timestamp x y z
        data = pd.read_csv(file_path, sep=" ", header=None, names=["timestamp", "x", "y", "z"])
        return data

    def generate_trajectory_plot(self, dataset_name):
        filename = f"St_{dataset_name}_Stereo_Akram.txt"
        data = self.load_trajectory(filename)
        
        if data is None:
            return

        # 1. Initialize a clean figure for your slides/report
        plt.figure(figsize=(9, 7))
        
        # 2. Plot the continuous trajectory line (Top-Down 2D Birds-Eye View)
        plt.plot(data["x"], data["z"], color="#1f77b4", linewidth=2.5, label="Estimated Stereo VIO Path", zorder=1)

        # 3. Extract boundary coordinates for Start and Stop markers
        start_x, start_z = data["x"].iloc[0], data["z"].iloc[0]
        stop_x, stop_z = data["x"].iloc[-1], data["z"].iloc[-1]

        # 4. Draw prominent Start and Stop Indicators
        plt.scatter(start_x, start_z, color="#2ca02c", marker="o", s=150, edgecolors="black", linewidths=1.5, label=f"START Point ({start_x:.2f}, {start_z:.2f})", zorder=2)
        plt.scatter(stop_x, stop_z, color="#d62728", marker="X", s=180, edgecolors="black", linewidths=1.5, label=f"STOP Point ({stop_x:.2f}, {stop_z:.2f})", zorder=2)

        # 5. Apply academic styling parameters
        plt.title(f"2D Trajectory Path Analysis\nSequence: {dataset_name}", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("X Axis Position (meters)", fontsize=11, labelpad=10)
        plt.ylabel("Z Axis Position (meters)", fontsize=11, labelpad=10)
        
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.axis("equal") # Crucial for 1:1 real-world physical proportions
        plt.legend(loc="best", fontsize=10, frameon=True, shadow=True)
        plt.tight_layout()

        # 6. Export high-resolution standalone visualization image
        save_path = os.path.join(self.output_dir, f"Final_Path_{dataset_name}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"📊 Trajectory graph saved for {dataset_name} -> {os.path.basename(save_path)}")


if __name__ == "__main__":
    plotter = VIOPlotter()
    
    target_sequences = [
        "dataset-room2_512_16",
        "dataset-corridor3_512_16",
        "dataset-outdoors5_512_16"
    ]
    
    print("🚀 Generating Final Trajectory Plots with Boundary Indicators...")
    for sequence in target_sequences:
        plotter.generate_trajectory_plot(sequence)
    print("\n🏁 All plots successfully generated! Check your exported_media/final_plots folder.")