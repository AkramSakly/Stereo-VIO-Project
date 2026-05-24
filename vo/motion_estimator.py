import cv2
import numpy as np
import os
from data_loader import image_dir, images
from feature_tracker import FeatureTracker

class MotionEstimator:
    def __init__(self):
        # TUM VI 512x512 Calibration
        self.K = np.array([[190.97, 0, 254.93],
                           [0, 190.97, 254.93],
                           [0, 0, 1]], dtype=np.float32)
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))

    def estimate_motion(self, prev_pts, curr_pts):
        if prev_pts is None or curr_pts is None or len(prev_pts) < 8:
            return self.cur_R, self.cur_t

        # Force correct types and shapes for OpenCV
        pts_prev = np.array(prev_pts, dtype=np.float32).reshape(-1, 2)
        pts_curr = np.array(curr_pts, dtype=np.float32).reshape(-1, 2)

        E, mask = cv2.findEssentialMat(pts_curr, pts_prev, self.K, 
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None or E.shape != (3, 3):
            return self.cur_R, self.cur_t

        _, R, t, mask = cv2.recoverPose(E, pts_curr, pts_prev, self.K, mask=mask)

        # Monocular VO Scale is unknown, using 1.0 as baseline
        absolute_scale = 1.0 
        self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
        self.cur_R = R.dot(self.cur_R)

        return self.cur_R, self.cur_t

if __name__ == "__main__":
    tracker = FeatureTracker()
    estimator = MotionEstimator()
    traj = np.zeros((600, 600, 3), dtype=np.uint8)

    for name in images:
        frame = cv2.imread(os.path.join(image_dir, name), cv2.IMREAD_GRAYSCALE)
        prev, curr = tracker.track(frame)
        
        if prev is not None and curr is not None:
            R, t = estimator.estimate_motion(prev, curr)
            
            # Map coordinates (Top-down view)
            draw_x = int(t[0][0]) + 300
            draw_y = int(t[2][0]) + 300
            cv2.circle(traj, (draw_x, draw_y), 1, (0, 255, 0), 2)
            
        cv2.imshow("Camera View", frame)
        cv2.imshow("Trajectory", traj)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()