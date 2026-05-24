import numpy as np
import cv2

class MonocularVO:

    def __init__(self, K):
        self.K = K
        self.pose = np.eye(4)

        # CORRIDOR OPTIMIZED FEATURES
        self.orb = cv2.ORB_create(
            nfeatures=8000,
            scaleFactor=1.1,
            nlevels=10,
            edgeThreshold=15
        )

    def process(self, img1, img2):

        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # improve corridor texture
        img1 = cv2.equalizeHist(img1)
        img2 = cv2.equalizeHist(img2)

        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:  # strict filter
                good.append(m)

        if len(good) < 12:
            return

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        E, _ = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=0.6
        )

        if E is None:
            return

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)

        if R is None or t is None:
            return

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t[:, 0]

        # correct SE(3) update
        self.pose = self.pose @ np.linalg.inv(T)