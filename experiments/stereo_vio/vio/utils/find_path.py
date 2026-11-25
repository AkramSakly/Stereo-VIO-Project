import cv2
import numpy as np
import pandas as pd
import os

class TumViLoader:
    def __init__(self, sequence_path):
        self.path = sequence_path
        self.imu_path = os.path.join(sequence_path, 'imu0', 'data.csv')
        self.cam0_path = os.path.join(sequence_path, 'cam0', 'data')
        
        if not os.path.exists(self.imu_path):
            raise FileNotFoundError(f"IMU data not found at: {self.imu_path}")
            
        self.imu_df = pd.read_csv(self.imu_path)
        self.cam_files = sorted(os.listdir(self.cam0_path))

    def get_frame_and_imu(self, idx):
        img = cv2.imread(os.path.join(self.cam0_path, self.cam_files[idx]))
        imu_slice = self.imu_df.iloc[idx*10 : (idx+1)*10]
        return img, imu_slice

class VIOEstimator:
    def __init__(self):
        self.state = np.zeros(3) # [x, y, z]
        self.prev_gray = None
        self.prev_pts = None

    def process(self, img, imu_batch):
        # Convert to grayscale for feature detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Initialization or Re-seeding (Preventing crashes)
        if self.prev_gray is None or (self.prev_pts is not None and len(self.prev_pts) < 10):
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=100, 
                                                    qualityLevel=0.3, minDistance=7)
            return self.state

        # 2. Visual Frontend: Optical Flow Tracking
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None)
        
        # Robustness check: if tracker fails, reset to re-seed next frame
        if curr_pts is None or status is None:
            self.prev_gray = None
            return self.state

        # 3. Motion Estimation
        good_new = curr_pts[status == 1]
        good_old = self.prev_pts[status == 1]
        
        if len(good_new) > 10: 
            displacement = np.mean(good_new - good_old, axis=0)
            self.state[0] += displacement[0] * 0.005 
            self.state[1] += displacement[1] * 0.005
            self.prev_pts = good_new.reshape(-1, 1, 2)
        else:
            # Not enough points, trigger re-seed
            self.prev_gray = None 

        self.prev_gray = gray.copy()
        
        return self.state

def main():
    # YOUR VERIFIED PATH
    dataset_path = r"C:\Users\selva\vio_project\data\raw\dataset-room2_512_16\mav0"
    
    try:
        loader = TumViLoader(dataset_path)
        vio = VIOEstimator()
        
        print(f"System ready. Processing frames from: {dataset_path}")

        for i in range(len(loader.cam_files) - 1):
            img, imu_batch = loader.get_frame_and_imu(i)
            
            # Run the estimator
            state = vio.process(img, imu_batch)
            
            # Output every 50 frames
            if i % 50 == 0:
                print(f"Frame {i:04d} | Est. Position: X={state[0]:.4f}, Y={state[1]:.4f}")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    main()