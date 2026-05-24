class TumViLoader:
    def __init__(self, sequence_path):
        self.path = sequence_path
        self.imu_path = os.path.join(sequence_path, 'imu0', 'data.csv')
        self.cam0_path = os.path.join(sequence_path, 'cam0', 'data')
        
        # Look for ground truth in standard TUM-VI location
        self.gt_path = os.path.join(sequence_path, 'state_groundtruth_estimate0', 'data.csv')
        
        self.imu_df = pd.read_csv(self.imu_path)
        self.cam_files = sorted(os.listdir(self.cam0_path))
        
        # ONLY load Ground Truth if it actually exists
        if os.path.exists(self.gt_path):
            self.gt_df = pd.read_csv(self.gt_path)
            print("Ground Truth loaded successfully.")
        else:
            self.gt_df = None
            print("WARNING: Ground Truth file not found. Skipping blue line plot.")

    def get_ground_truth(self):
        if self.gt_df is not None:
            return self.gt_df.iloc[:, 1].values, self.gt_df.iloc[:, 2].values
        return None, None

# --- In your main() function, update the plotting logic ---

def main():
    dataset_path = r"C:\Users\selva\vio_project\data\raw\dataset-room2_512_16\mav0"
    
    loader = TumViLoader(dataset_path)
    vio = VIOEstimator()
    gt_x, gt_y = loader.get_ground_truth()
    
    path_x, path_y = [], []

    # ... [Looping code remains the same] ...

    # --- UPDATED PLOTTING LOGIC ---
    plt.figure(figsize=(10, 6))
    
    # Only plot blue line IF ground truth was found
    if gt_x is not None:
        plt.plot(gt_x, gt_y, 'b-', label='Ground Truth', linewidth=1.5, alpha=0.6)
    
    x_smooth = smooth_path(path_x)
    y_smooth = smooth_path(path_y)
    plt.plot(x_smooth, y_smooth, 'r--', label='Your VO Estimation', linewidth=1.5)
    
    plt.legend()
    plt.grid(True)
    plt.show()