import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

class defense_plot_engine:
    def __init__(self, base_path=r"C:\Users\selva\vio_project"):
        self.base_path = base_path
        self.output_dir = os.path.join(self.base_path, "exported_media", "defense_plots")
        
        # Ensure output directory exists for presentation-ready media
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 Created defense plots folder at: {self.output_dir}")

    def load_trace_file(self, filename):
        file_path = os.path.join(self.base_path, filename)
        if not os.path.exists(file_path):
            print(f"⚠️ Trajectory trace missing: {filename}")
            return None
        
        # Parse standard VIO output format: timestamp(ns) x y z
        data = pd.read_csv(file_path, sep=" ", header=None, names=["t", "x", "y", "z"])
        return data

    def generate_trajectory_figure(self, dataset_id, sequence_title):
        filename = f"St_{dataset_id}_Stereo_Akram.txt"
        data = self.load_trace_file(filename)
        
        if data is None:
            return

        # 1. Initialize professional academic figure canvas (Top-Down X-Z View)
        plt.figure(figsize=(10, 8))
        
        # 2. Plot the continuous trajectory path (Blue Line)
        plt.plot(data["x"], data["z"], color="#1f77b4", linewidth=2.8, label="Estimated Stereo VIO Path", zorder=1)

        # 3. Extract boundary coordinates for Start and Stop Markers
        start_x, start_z = data["x"].iloc[0], data["z"].iloc[0]
        stop_x, stop_z = data["x"].iloc[-1], data["z"].iloc[-1]

        # 4. Burn prominent START/STOP markers into the plot
        # Green Circle for Start | Red X for Stop
        plt.scatter(start_x, start_z, color="#2ca02c", marker="o", s=200, edgecolors="black", linewidths=1.5, label="START (MoCap Initialization)", zorder=2)
        plt.scatter(stop_x, stop_z, color="#d62728", marker="X", s=220, edgecolors="black", linewidths=1.5, label="STOP (Final Frame State)", zorder=2)

        # 5. Apply Academic Styling Parameters matching benchmark aesthetics
        plt.title(f"Technical VIO Performance Analysis\n{sequence_title}", fontsize=14, fontweight="bold", pad=20)
        plt.xlabel("X Position (meters)", fontsize=11, labelpad=12)
        plt.ylabel("Z Position (meters)", fontsize=11, labelpad=12)
        
        plt.grid(True, linestyle="--", alpha=0.6, zorder=0)
        plt.axis("equal") # Crucial for accurate structural proportions
        plt.legend(loc="best", fontsize=11, frameon=True, shadow=True, fancybox=True)
        plt.tight_layout()

        # 6. Export high-resolution standalone visualization image
        save_path = os.path.join(self.output_dir, f"Final_Schematic_{dataset_id}.png")
        plt.savefig(save_path, dpi=300) # 300 DPI for print/presentation quality
        plt.close()
        print(f"📊 Schematic figure generated for {dataset_id} -> {os.path.basename(save_path)}")


if __name__ == "__main__":
    # Test setting targeting your mandatory evaluation directories
    engine = defense_plot_engine()
    
    target_sequences = [
        ("dataset-room2_512_16", "ROOM 2 (Closed-Loop Office Tracking)"),
        ("dataset-corridor3_512_16", "CORRIDOR 3 (Straight Structural Hallway)"),
        ("dataset-outdoors5_512_16", "OUTDOORS 5 (Large Campus Courtyard Loop)")
    ]
    
    print("🚀 Generating academic-grade defense schematics for all sequences...")
    for ds_id, title in target_sequences:
        engine.generate_trajectory_figure(ds_id, title)
    print("\n🏁 All schematic figures successfully generated! Check your exported_media/defense_plots folder.")