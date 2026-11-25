import pandas as pd
import numpy as np
import os

class VIOBuffer:
    def __init__(self, imu_file_path=None):
        self.imu_data = None
        if imu_file_path:
            self.load_data(imu_file_path)

    def load_data(self, imu_file_path):
        if not os.path.exists(imu_file_path):
            print(f"❌ Buffer Error: File not found at {imu_file_path}")
            return
        
        # TUM-VI format: #timestamp [ns], w_x, w_y, w_z, a_x, a_y, a_z
        # Using engine='python' and sep=',' to be safe with CSV formats
        self.imu_data = pd.read_csv(imu_file_path)
        print(f"✅ Buffer: Loaded {len(self.imu_data)} IMU measurements.")

    def get_measurements(self, t_start, t_end):
        """Returns IMU data between two camera frame timestamps."""
        if self.imu_data is None:
            return None
        mask = (self.imu_data.iloc[:, 0] >= t_start) & (self.imu_data.iloc[:, 0] <= t_end)
        return self.imu_data.loc[mask].values

# --- REMOVE OR COMMENT OUT THE TEST CODE AT THE BOTTOM ---
# if __name__ == "__main__":
#    # Do not import data_loader here!
#    pass