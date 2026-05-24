import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_and_plot(csv_path):
    if not os.path.exists(csv_path):
        print(f"🟡 Skipping: {csv_path} (File not found yet)")
        return
        
    # Load the data
    df = pd.read_csv(csv_path)
    dataset_name = os.path.basename(csv_path)

    # 1. CALCULATE METRICS (For your Report)
    # Distance between each consecutive point
    dx = np.diff(df['x'])
    dy = np.diff(df['y'])
    dz = np.diff(df['z'])
    step_distances = np.sqrt(dx**2 + dy**2 + dz**2)
    total_dist = np.sum(step_distances)
    
    # Drift = Distance from origin at the final frame
    final_error = np.sqrt(df['x'].iloc[-1]**2 + df['y'].iloc[-1]**2 + df['z'].iloc[-1]**2)
    drift_rate = (final_error / total_dist) * 100 if total_dist > 0 else 0

    print(f"\n✅ Analysis for: {dataset_name}")
    print(f"   📏 Total Trajectory Length: {total_dist:.2f} meters")
    print(f"   ⚠️ Final Drift Error: {final_error:.2f} meters")
    print(f"   📉 Drift Rate: {drift_rate:.2f}%")

    # 2. CREATE THE PLOTS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot A: Top-Down Path (X vs Z)
    ax1.plot(df['x'], df['z'], color='blue', label='VIO Path')
    ax1.scatter(df['x'].iloc[0], df['z'].iloc[0], c='green', marker='o', s=100, label='Start')
    ax1.scatter(df['x'].iloc[-1], df['z'].iloc[-1], c='red', marker='x', s=100, label='End')
    ax1.set_title(f"Top-Down Trajectory (Bird's Eye)\n{dataset_name}")
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Z (meters)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot B: Vertical Drift (Y over Time)
    time_axis = np.arange(len(df)) # Using frame index as proxy for time
    ax2.plot(time_axis, df['y'], color='purple')
    ax2.set_title("Vertical Stability (Y-axis Drift)")
    ax2.set_xlabel("Frame Number")
    ax2.set_ylabel("Height Y (meters)")
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = csv_path.replace('.csv', '_analysis.png')
    plt.savefig(save_path)
    print(f"   🖼️ Plot saved to: {save_path}")
    plt.show()

# --- RUN FOR YOUR DATASETS ---
analyze_and_plot('results_stereo_dataset-corridor3_512_16.csv')
analyze_and_plot('results_stereo_dataset-outdoors5_512_16.csv')