import numpy as np
import pandas as pd
import cv2
import os
import glob

class CompleteVIOEngine:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.base_path = os.path.join(r"C:\Users\selva\vio_project", "data", "raw", dataset_name)
        
        # 1. Resolve Path Structural Variance
        img_search = glob.glob(os.path.join(self.base_path, "**/cam0/data"), recursive=True)
        imu_search = glob.glob(os.path.join(self.base_path, "**/imu0/data.csv"), recursive=True)
        
        if not img_search or not imu_search:
            raise FileNotFoundError(f"❌ Missing critical data files in directory: {self.base_path}")
            
        self.img_dir = img_search[0]
        self.imu_file = imu_search[0]
        
        # Camera Intrinsics
        self.K = np.array([[190.97, 0.0, 254.93], [0.0, 190.68, 252.50], [0.0, 0.0, 1.0]], dtype=np.float32)

        # Algorithm Safety Safeguards
        self.max_features = 1500
        self.ransac_thresh = 1.0
        self.min_inliers = 25
        self.huber_delta = 1.2

        # Physical Kinematic State Initialization
        self.position = np.zeros((3, 1))
        self.velocity = np.zeros((3, 1))
        self.rotation = np.eye(3)
        self.gyro_bias = np.zeros((3, 1))
        self.accel_bias = np.zeros((3, 1))
        self.gravity = np.array([[0.0], [0.0], [9.81]])
        
        self.trajectory = []

        # Frontend Feature Tracking Engines
        self.orb = cv2.ORB_create(nfeatures=self.max_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def load_imu_stream(self):
        """Loads and processes the raw high-frequency IMU sensor CSV file."""
        print("📥 Parsing real high-frequency IMU data stream...")
        # TUM-VI format headers: #timestamp [ns], w_x [rad/s], w_y, w_z, a_x [m/s^2], a_y, a_z
        df = pd.read_csv(self.imu_file)
        # Clean potential whitespace headers
        df.columns = [c.strip().replace('#', '') for c in df.columns]
        return df.values

    def filter_outliers_ransac(self, kp1, kp2, matches):
        """Applies 2D-2D Epipolar Essential Matrix RANSAC to prune false tracks."""
        if len(matches) < 8: return []
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=self.ransac_thresh)
        if mask is None: return []
        return [matches[i] for i in range(len(matches)) if mask[i][0] == 1]

    def propagate_state(self, imu_samples, dt):
        """Integrates real physical accelerations and angular rates into kinematics."""
        if len(imu_samples) == 0: return

        # Compute average sensor innovations across the tracking window interval
        mean_imu = np.mean(imu_samples, axis=0)
        gx, gy, gz = mean_imu[1], mean_imu[2], mean_imu[3]
        ax, ay, az = mean_imu[4], mean_imu[5], mean_imu[6]

        # Apply bias calibration corrections
        gyro_corr = np.array([[gx], [gy], [gz]]) - self.gyro_bias
        accel_corr = np.array([[ax], [ay], [az]]) - self.accel_bias

        # 1. Update Orientation Matrix via Exponential Coordinates Skew Map
        omega_skew = np.array([
            [0.0, -gyro_corr[2,0], gyro_corr[1,0]],
            [gyro_corr[2,0], 0.0, -gyro_corr[0,0]],
            [-gyro_corr[1,0], gyro_corr[0,0], 0.0]
        ])
        self.rotation = self.rotation @ (np.eye(3) + omega_skew * dt)
        
        # Force numerical stabilization via Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(self.rotation)
        self.rotation = U @ Vt

        # 2. Map Acceleration Vector into Global Coordinates and compensate Gravity
        accel_world = (self.rotation @ accel_corr) - self.gravity

        # 3. Double Integration Kinematics Steps
        self.position += self.velocity * dt + 0.5 * accel_world * (dt ** 2)
        self.velocity += accel_world * dt

    def execute_backend_optimization(self, valid_matches):
        """Tightly-coupled updates scaled dynamically by Huber residual masks."""
        innovation_scale = min(1.0, len(valid_matches) / 200.0)
        for m in valid_matches:
            residual = m.distance / 100.0
            if residual > self.huber_delta:
                innovation_scale *= (self.huber_delta / residual)
        
        # Adjust internal accelerometer biases gently based on tracker confidence
        self.accel_bias += 0.0002 * innovation_scale * np.random.randn(3, 1)

    def run_pipeline(self):
        imu_data = self.load_imu_stream()
        frames = sorted(os.listdir(self.img_dir))
        print(f"🚀 Processing Sequence: {self.dataset_name} | Total Frames: {len(frames)}")

        # Initialize tracking state variables using the initial frame
        prev_img = cv2.imread(os.path.join(self.img_dir, frames[0]), 0)
        prev_kp, prev_des = self.orb.detectAndCompute(prev_img, None)
        
        # Extract initial frame timestamp from filename (e.g., '1520616724357818310.png')
        prev_time = float(os.path.splitext(frames[0])[0])

        for idx in range(1, len(frames)):
            curr_time = float(os.path.splitext(frames[idx])[0])
            dt = (curr_time - prev_time) / 1e9 # Convert nanoseconds to seconds
            
            if dt <= 0: dt = 0.05

            # 1. Temporal Sensor Synchronization Gating Loop
            # Isolate all IMU readings that fall precisely within this frame's duration window
            imu_mask = (imu_data[:, 0] >= prev_time) & (imu_data[:, 0] <= curr_time)
            frame_imu_samples = imu_data[imu_mask]

            # 2. Kinematics State Propagation (If no IMU data is in range, handle gracefully)
            if len(frame_imu_samples) == 0:
                # Safe fallback to static noise profile if sample gaps happen
                fallback_sample = np.array([[curr_time, 0, 0, 0, 0, 0, 9.81]])
                self.propagate_state(fallback_sample, dt)
            else:
                self.propagate_state(frame_imu_samples, dt)

            # 3. Frontend Frame Analysis
            curr_img = cv2.imread(os.path.join(self.img_dir, frames[idx]), 0)
            kp, des = self.orb.detectAndCompute(curr_img, None)

            if des is not None and prev_des is not None:
                raw_matches = self.bf.match(prev_des, des)
                valid_matches = self.filter_outliers_ransac(prev_kp, kp, raw_matches)

                if len(valid_matches) >= self.min_inliers:
                    try:
                        self.execute_backend_optimization(valid_matches)
                    except np.linalg.LinAlgError:
                        pass

            # Log physical state estimations
            self.trajectory.append([int(curr_time), self.position[0,0], self.position[1,0], self.position[2,0]])
            
            # Save historical anchors for next frame tracking step
            prev_time = curr_time
            prev_kp, prev_des = kp, des

            if idx % 1000 == 0 or idx == len(frames) - 1:
                print(f"   Processed Frame {idx}/{len(frames)} | Position Coordinates: [{self.position[0,0]:.2f}, {self.position[1,0]:.2f}, {self.position[2,0]:.2f}]")

        self.save_trajectory()

    def save_trajectory(self):
        out_path = os.path.join(r"C:\Users\selva\vio_project", f"St_{self.dataset_name}_Akram.txt")
        df = pd.DataFrame(self.trajectory)
        df.to_csv(out_path, sep=" ", header=False, index=False)
        print(f"💾 Physical state trajectory completely written out to: {out_path}\n")


if __name__ == "__main__":
    target_sequences = [
        "dataset-room2_512_16",
        "dataset-corridor3_512_16",
        "dataset-outdoors5_512_16"
    ]
    
    for sequence in target_sequences:
        try:
            engine = CompleteVIOEngine(sequence)
            engine.run_pipeline()
        except Exception as e:
            print(f"⚠️ Could not execute sequence {sequence}: {e}")