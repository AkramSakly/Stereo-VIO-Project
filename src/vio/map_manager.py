import numpy as np
import cv2

class MapManager:
    def __init__(self, K):
        self.K = K
        self.landmarks_3d = []

    def triangulate_new_points(self, R1, t1, R2, t2, pts1, pts2):
        # Create Projection Matrices P = K[R|t]
        P1 = self.K @ np.hstack((R1, t1))
        P2 = self.K @ np.hstack((R2, t2))
        
        # Reshape for OpenCV (2, N)
        p1 = pts1.reshape(-1, 2).T
        p2 = pts2.reshape(-1, 2).T
        
        # Triangulate
        pts4d = cv2.triangulatePoints(P1, P2, p1, p2)
        pts3d = pts4d[:3, :] / pts4d[3, :]
        
        return pts3d.T

    def add_keyframe(self, R, t, points):
        if points is not None:
            self.landmarks_3d.extend(points)
        # Keep map size manageable
        if len(self.landmarks_3d) > 1000:
            self.landmarks_3d = self.landmarks_3d[-1000:]