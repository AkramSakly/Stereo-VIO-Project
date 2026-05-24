import cv2
import numpy as np
import os
from data_loader import image_dir, images

class FeatureTracker:
    def __init__(self):
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        self.prev_img = None
        self.prev_pts = None

    def detect_features(self, img):
        kp = self.fast.detect(img, None)
        return np.array([p.pt for p in kp], dtype=np.float32).reshape(-1, 1, 2)

    def track(self, img):
        if self.prev_img is None:
            self.prev_pts = self.detect_features(img)
            self.prev_img = img
            return None, self.prev_pts

        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_img, img, self.prev_pts, None, **self.lk_params)
        
        status = status.reshape(-1)
        valid_prev = self.prev_pts[status == 1]
        valid_curr = curr_pts[status == 1]

        if len(valid_curr) < 100:
            valid_curr = self.detect_features(img)
            valid_prev = None 

        self.prev_img = img
        self.prev_pts = valid_curr.reshape(-1, 1, 2)
        
        return valid_prev, valid_curr

if __name__ == "__main__":
    tracker = FeatureTracker()
    for name in images:
        frame = cv2.imread(os.path.join(image_dir, name), cv2.IMREAD_GRAYSCALE)
        _, curr = tracker.track(frame)
        display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if curr is not None:
            for p in curr:
                x, y = p.ravel()
                cv2.circle(display, (int(x), int(y)), 2, (0, 255, 0), -1)
        cv2.imshow("Tracker", display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()