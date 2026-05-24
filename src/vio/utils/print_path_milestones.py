import os
import pandas as pd
import numpy as np

class PathMilestonePrinter:
    def __init__(self, base_path=r"C:\Users\selva\vio_project"):
        self.base_path = base_path
        self.target_sequences = [
            ("dataset-room2_512_16", "ROOM 2 (MoCap Office Enclosure)"),
            ("dataset-corridor3_512_16", "CORRIDOR 3 (Long Structural Hallway)"),
            ("dataset-outdoors5_512_16", "OUTDOORS 5 (Open Air Campus Courtyard)")
        ]

    def process_and_print(self):
        print("=" * 80)
        print("📊 TUM-VI BENCHMARK PHYSICAL TRAJECTORY MILESTONES (STEREO CORE)")
        print("=" * 80)

        for seq_id, descriptions in self.target_sequences:
            filename = f"St_{seq_id}_Stereo_Akram.txt"
            file_path = os.path.join(self.base_path, filename)

            if not os.path.exists(file_path):
                print(f"\n⚠️ Missing trace file for: {descriptions}")
                print(f"   Expected location: {filename}")
                continue

            # Load the space coordinates (timestamp x y z)
            data = pd.read_csv(file_path, sep=" ", header=None, names=["t", "x", "y", "z"])
            total_points = len(data)

            # Extract distinct physical milestones along the path array
            start_pt = data.iloc[0]
            mid_pt = data.iloc[total_points // 2]
            stop_pt = data.iloc[-1]

            # Calculate total approximate physical displacement bounds (Max dimensions)
            x_range = data["x"].max() - data["x"].min()
            z_range = data["z"].max() - data["z"].min()
            y_drift = data["y"].max() - data["y"].min()

            print(f"\n🎬 SEQUENCE: {descriptions}")
            print(f"   ┌─ Total Logged Tracking States : {total_points} positions")
            print(f"   ├─ Real-World Boundary Enclosure: Length (ΔX) = {x_range:.2f}m | Width (ΔZ) = {z_range:.2f}m")
            print(f"   ├─ Vertical Amplitude Window    : Height (ΔY) = {y_drift:.2f}m")
            print(f"   │")
            print(f"   ├─ 📍 START (Frame 0 Initial)   : [ X: {start_pt['x']:>6.2f}m , Y: {start_pt['y']:>6.2f}m , Z: {start_pt['z']:>6.2f}m ]")
            print(f"   ├─ 📍 MID-WAY (Halfway Turn)     : [ X: {mid_pt['x']:>6.2f}m , Y: {mid_pt['y']:>6.2f}m , Z: {mid_pt['z']:>6.2f}m ]")
            print(f"   └─ 📍 STOP (Final Settled Frame): [ X: {stop_pt['x']:>6.2f}m , Y: {stop_pt['y']:>6.2f}m , Z: {stop_pt['z']:>6.2f}m ]")
            print("-" * 80)

if __name__ == "__main__":
    printer = PathMilestonePrinter()
    printer.process_and_print()