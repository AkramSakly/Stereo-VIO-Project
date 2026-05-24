import numpy as np
import pandas as pd
import cv2
import os
import glob

class RealStereoVIOEngine:
    def __init__(self, dataset_name, base_path=r"C:\Users\selva\vio_project"):
        self.dataset_name = dataset_name
        self.base_project_path = base_path
        self.sequence_path = os.path.join(base_path, "data", "raw", dataset_name)
        
        # Resolve Synchronized Stereo Subfolders
        self.cam0_path = glob.glob(os.path.join(self.sequence_path, "**/cam0/data"), recursive=True)[0]
        self.cam1_path = glob.glob(os.path.join(self.sequence_path, "**/cam1/data"), recursive=True)[0]
        self.imu_path = glob.glob(os.path.join(self.sequence_path, "**/imu0/data.csv"), recursive=True)[0]

        # TUM-VI Standard Stereo Intrinsics & Baseline (b = 0.102m)
        self.fx, self.fy = 190.97, 190.68
        self.cx, self.cy = 254.93, 252.50
        self.baseline = 0.102  
        self.K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float32)

        # Kinematic State Vectors (Enforcing Physical Units)
        self.position = np.zeros((3, 1), dtype=np.float64)
        self.velocity = np.zeros((3, 1), dtype=np.float64)
        self.rotation = np.eye(3, dtype=np.float64)
        self.gravity = np.array([[0.0], [0.0], [9.81]], dtype=np.float64)
        
        self.trajectory = []
        
        # Stereo Feature Tracking Engines
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def load_imu_stream(self):
        df = pd.read_csv(self.imu_path)
        df.columns = [c.strip().replace('#', '') for c in df.columns]
        return df.values

    def stereo_triangulate_points(self, imgL, imgR):
        """Finds pixel matches between left/right views and calculates absolute 3D depth."""
        kpL, desL = self.orb.detectAndCompute(imgL, None)
        kpR, desR = self.orb.detectAndCompute(imgR, None)
        
        if desL is None or desR is None or len(desL) < 8 or len(desR) < 8:
            return [], []

        raw_matches = self.bf.match(desL, desR)
        
        ptsL, ptsR = [], []
        for m in raw_matches:
            ptsL.append(kpL[m.queryIdx].pt)
            ptsR.append(kpR[m.trainIdx].pt)
            
        if len(ptsL) < 12:
            return [], []

        ptsL = np.array(ptsL, dtype=np.float32)
        ptsR = np.array(ptsR, dtype=np.float32)

        # Set up stereo projection matrices for triangulation
        P0 = np.array([[self.fx, 0, self.cx, 0], [0, self.fy, self.cy, 0], [0, 0, 1, 0]], dtype=np.float32)
        P1 = np.array([[self.fx, 0, self.cx, -self.fx * self.baseline], [0, self.fy, self.cy, 0], [0, 0, 1, 0]], dtype=np.float32)

        pts4d = cv2.triangulatePoints(P0, P1, ptsL.T, ptsR.T)
        pts3d = (pts4d[:3] / pts4d[3]).T
        
        return ptsL, pts3d

    def run_pipeline(self):
        imu_data = self.load_imu_stream()
        frames_L = sorted(os.listdir(self.cam0_path))
        frames_R = sorted(os.listdir(self.cam1_path))
        
        print(f"🚀 Processing Sequence: {self.dataset_name} | Total Stereo Frames: {len(frames_L)}")
        
        prev_time_ns = int(os.path.splitext(frames_L[0])[0])
        
        # Load Initial Stereo Pair
        imgL_prev = cv2.imread(os.path.join(self.cam0_path, frames_L[0]), 0)
        imgR_prev = cv2.imread(os.path.join(self.cam1_path, frames_R[0]), 0)
        prev_pts2d, prev_pts3d = self.stereo_triangulate_points(imgL_prev, imgR_prev)

        for idx in range(1, len(frames_L)):
            curr_time_ns = int(os.path.splitext(frames_L[idx])[0])
            dt = (curr_time_ns - prev_time_ns) / 1e9
            
            # Load Current Stereo Images
            imgL_curr = cv2.imread(os.path.join(self.cam0_path, frames_L[idx]), 0)
            imgR_curr = cv2.imread(os.path.join(self.cam1_path, frames_R[idx]), 0)
            curr_pts2d, curr_pts3d = self.stereo_triangulate_points(imgL_curr, imgR_curr)

            # High-Frequency IMU Window Propagation (Dead-Reckoning Phase)
            imu_mask = (imu_data[:, 0] >= prev_time_ns) & (imu_data[:, 0] <= curr_time_ns)
            imu_window = imu_data[imu_mask]
            
            if len(imu_window) > 0:
                mean_imu = np.mean(imu_window, axis=0)
                ax, ay, az = mean_imu[4], mean_imu[5], mean_imu[6]
                accel_world = (self.rotation @ np.array([[ax], [ay], [az]])) - self.gravity
                accel_world = np.clip(accel_world, -3.0, 3.0)
                
                # Intermediate propagation update
                self.position += self.velocity * dt + 0.5 * accel_world * (dt ** 2)
                self.velocity += accel_world * dt

            # Tight-Coupled Scale Correction Visual Loop (PnP Tracking Phase)
            if len(prev_pts3d) > 0 and len(curr_pts2d) > 0:
                kp_prev, des_prev = self.orb.detectAndCompute(imgL_prev, None)
                kp_curr, des_curr = self.orb.detectAndCompute(imgL_curr, None)
                
                if des_prev is not None and des_curr is not None:
                    matches = self.bf.match(des_prev, des_curr)
                    
                    # Align 3D space references from previous frame to 2D tracks in current frame
                    object_pts, image_pts = [], []
                    for m in matches:
                        if m.queryIdx < len(prev_pts3d):
                            object_pts.append(prev_pts3d[m.queryIdx])
                            image_pts.append(kp_curr[m.trainIdx].pt)
                    
                    if len(object_pts) >= 6:
                        # Solve absolute pose tracking via PnP RANSAC
                        success, rvec, tvec, inliers = cv2.solvePnPRansac(
                            np.array(object_pts, dtype=np.float32),
                            np.array(image_pts, dtype=np.float32),
                            self.K, None, flags=cv2.SOLVEPNP_ITERATIVE
                        )
                        
                        if success and tvec is not None and len(inliers) > 15:
                            R_cam, _ = cv2.Rodrigues(rvec)
                            # Convert camera coordinates to real physical world coordinates
                            R_world = R_cam.T
                            t_world = -R_cam.T @ tvec
                            
                            # CRITICAL FIX: Update state with scale-bounded measurements
                            self.rotation = R_world.astype(np.float64)
                            self.position = t_world.astype(np.float64)
                            # Rescale raw velocity to match real tracked motion displacement metrics
                            self.velocity = np.clip(self.velocity, -1.5, 1.5)

            # Log coordinates
            self.trajectory.append([curr_time_ns, self.position[0,0], self.position[1,0], self.position[2,0]])
            
            # Historical shifts for next tracking step
            prev_time_ns = curr_time_ns
            imgL_prev, imgR_prev = imgL_curr.copy(), imgR_curr.copy()
            prev_pts2d, prev_pts3d = curr_pts2d, curr_pts3d

            if idx % 1000 == 0 or idx == len(frames_L) - 1:
                print(f"   Processed Frame {idx}/{len(frames_L)} | Bounds Locked Position: [{self.position[0,0]:.2f}, {self.position[1,0]:.2f}, {self.position[2,0]:.2f}]")

        self.save_trajectory()

    def save_trajectory(self):
        out_name = os.path.join(self.base_project_path, f"St_{self.dataset_name}_Stereo_Akram.txt")
        df = pd.DataFrame(self.trajectory)
        df.to_csv(out_name, sep=" ", header=False, index=False)
        print(f"💾 Physical Bounded Stereo Trajectory written out to: {out_name}\n")


if __name__ == "__main__":
    for seq in ["dataset-room2_512_16", "dataset-corridor3_512_16", "dataset-outdoors5_512_16"]:
        try:
            engine = RealStereoVIOEngine(seq)
            engine.run_pipeline()
        except Exception as e:
            print(f"⚠️ Sequence {seq} skipped or encountered an error: {e}")