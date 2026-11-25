import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class StereoTrajectoryPlotter:
    def __init__(self, base_path=r"C:\Users\selva\vio_project"):
        self.base_path = base_path
        self.output_dir = os.path.join(self.base_path, "exported_media", "plots")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def parse_trajectory(self, filename):
        file_path = os.path.join(self.base_path, filename)
        if not os.path.exists(file_path):
            print(f"⚠️ Trace file not found: {filename}")
            return None
        
        # TUM-VI format: timestamp(ns) x y z
        data = pd.read_csv(file_path, sep=" ", header=None, names=["timestamp", "x", "y", "z"])
        return data

    def plot_sequence(self, dataset_name):
        filename = f"St_{dataset_name}_Stereo_Akram.txt"
        data = self.parse_trajectory(filename)
        
        if data is None:
            return

        # Create a professional dual-subplot layout for your report/defense
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Stereo VIO Trajectory Analysis: {dataset_name}", fontsize=14, fontweight='bold')

        # Subplot 1: Top-Down X-Z Bird's-Eye View Geometry
        ax1.plot(data["x"], data["z"], color="blue", linewidth=2, label="Stereo VIO Path")
        ax1.scatter(data["x"].iloc[0], data["z"].iloc[0], color="green", marker="o", s=100, label="Start Location")
        ax1.scatter(data["x"].iloc[-1], data["z"].iloc[-1], color="red", marker="x", s=100, label="End Location")
        ax1.set_title("Top-Down Trajectory Geometry (X-Z Plane)")
        ax1.set_xlabel("X Position (meters)")
        ax1.set_ylabel("Z Position (meters)")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.axis("equal")
        ax1.legend()

        # Subplot 2: Altitude Elevation Stability over Timeline
        # Convert nanosecond timestamps to relative seconds for clean presentation
        time_seconds = (data["timestamp"] - data["timestamp"].iloc[0]) / 1e9
        ax2.plot(time_seconds, data["y"], color="darkorange", linewidth=2, label="Height (Y)")
        ax2.set_title("Altitude Elevation Profile over Time")
        ax2.set_xlabel("Relative Timeline (seconds)")
        ax2.set_ylabel("Y Altitude (meters)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        
        # Save the crisp diagnostic figure directly to your media directory
        output_plot_path = os.path.join(self.output_dir, f"Stereo_Analysis_{dataset_name}.png")
        plt.savefig(output_plot_path, dpi=300)
        plt.close()
        print(f"📊 Crisp analysis plot successfully saved to:\n   👉 {output_plot_path}")


if __name__ == "__main__":
    plotter = StereoTrajectoryPlotter()
    
    mandatory_sequences = [
        "dataset-room2_512_16",
        "dataset-corridor3_512_16",
        "dataset-outdoors5_512_16"
    ]
    
    for seq in mandatory_sequences:
        plotter.plot_sequence(seq)