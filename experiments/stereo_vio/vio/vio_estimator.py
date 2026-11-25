import cv2
import numpy as np
import os
import pandas as pd
import argparse
from vio_backend import VIOBackend

class VIOEstimator:
    def __init__(self, data_path):
        self.data_path = data_path
        self.detector = cv2.ORB_create(nfeatures=1200)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # TUM VI Intrinsics for the Le Creusot Lab datasets
        self.K = np.array([[190.97, 0, 254.93], 
                           [0, 190.97, 256.89], 
                           [0, 0, 1]], dtype=np.float32)
        
        self.prev_kp, self.prev_des = None, None
        self.backend = VIOBackend(window_size=10)
        self.position = np.zeros((3, 1))

    def get_pnp_anchor(self, frame):
        """Calculates the visual reference point using PnP."""
        kp, des = self.detector.detectAndCompute(frame, None)
        if self.prev_des is None or des is None:
            self.prev_kp, self.prev_des = kp, des
            return None

        matches = self.matcher.match(self.prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)[:60]

        if len(matches) < 15:
            return None

        pts_3d, pts_2d = [], []
        for m in matches:
            u, v = self.prev_kp[m.queryIdx].pt
            z = 2.5 # Approximate depth for visual tethering
            x = (u - self.K[0,2]) * z / self.K[0,0]
            y = (v - self.K[1,2]) * z / self.K[1,1]
            pts_3d.append([x, y, z])
            pts_2d.append(kp[m.trainIdx].pt)

        success, rvec, tvec = cv2.solvePnP(np.array(pts_3d, dtype=np.float32), 
                                           np.array(pts_2d, dtype=np.float32), 
                                           self.K, None)
        self.prev_kp, self.prev_des = kp, des
        return tvec if success else None

    def run(self, output_name):
        base_path = os.path.join(self.data_path, 'mav0')
        img_folder = os.path.join(base_path, "cam0", "data")
        imu_file = os.path.join(base_path, "imu0", "data.csv")
        
        imu_df = pd.read_csv(imu_file)
        img_list = sorted(os.listdir(img_folder))
        last_ts = int(img_list[0].replace(".png", ""))
        trajectory = []

        print(f"🚀 Starting VIO Optimization for: {output_name}")

        for i in range(len(img_list)):
            curr_ts = int(img_list[i].replace(".png", ""))
            dt = (curr_ts - last_ts) / 1e9
            img = cv2.imread(os.path.join(img_folder, img_list[i]), 0)

            # 1. Visual Tracking (PnP)
            v_anchor = self.get_pnp_anchor(img)

            # 2. Extract IMU segment
            mask = (imu_df.iloc[:,0] > last_ts) & (imu_df.iloc[:,0] <= curr_ts)
            imu_segment = imu_df[mask].iloc[:, 1:4].values # Acceleration X, Y, Z
            
            # 3. Backend Optimization (Week 10 Logic)
            if len(imu_segment) > 0:
                self.position = self.backend.launch_optimization(v_anchor, imu_segment, dt)

            # 4. Save in TUM Format
            trajectory.append(f"{curr_ts/1e9} {self.position[0,0]} {self.position[1,0]} {self.position[2,0]} 0 0 0 1")
            last_ts = curr_ts
            
            if i % 1000 == 0:
                print(f"Processed frame {i}. Drift corrected. Height: {self.position[2,0]:.2f}m")

        with open(output_name, "w") as f:
            f.write("\n".join(trajectory))
        print(f"✅ Success! VIO Trajectory saved to {output_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    dataset_path = os.path.join("data", "raw", args.dataset)
    VIOEstimator(dataset_path).run(f"St_{args.dataset}_Akram.txt")