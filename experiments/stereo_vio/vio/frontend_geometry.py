import numpy as np
import cv2
import os

class VisualFrontend:
    def __init__(self, feature_type="ORB", max_features=1500):
        self.feature_type = feature_type.upper()
        self.max_features = max_features
        
        # TUM-VI Standard Intrinsic Calibration Matrix for Cam0
        self.K = np.array([
            [190.97,   0.0,  254.93],
            [  0.0,  190.68, 252.50],
            [  0.0,    0.0,    1.0]
        ], dtype=np.float32)

        # 1. Initialize Feature Detector and Descriptor Extractors
        if self.feature_type == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=self.max_features)
            # SIFT uses L2 Norm for floating-point descriptors
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else: # Default to ORB
            self.detector = cv2.ORB_create(nfeatures=self.max_features)
            # ORB uses Hamming Distance for binary string descriptors
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def extract_features(self, frame):
        """Detects keypoints and computes their corresponding descriptors."""
        if frame is None:
            return [], None
        kp, des = self.detector.detectAndCompute(frame, None)
        return kp, des

    def match_frames(self, kp1, des1, kp2, des2, ratio_threshold=0.75):
        """
        Matches consecutive frames using a rigorous combination of:
        1. KNN Matching (k=2)
        2. Lowe's Ratio Test
        3. Bidirectional Cross-Checking Validation
        """
        if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
            return []

        # --- Step A: Forward Matching (Frame 1 -> Frame 2) ---
        matches_fwd = self.matcher.knnMatch(des1, des2, k=2)
        good_fwd = []
        for m_match in matches_fwd:
            if len(m_match) == 2:
                m, n = m_match
                if m.distance < ratio_threshold * n.distance:
                    good_fwd.append(m)

        # --- Step B: Backward Matching (Frame 2 -> Frame 1) ---
        matches_bwd = self.matcher.knnMatch(des2, des1, k=2)
        good_bwd = []
        for m_match in matches_bwd:
            if len(m_match) == 2:
                m, n = m_match
                if m.distance < ratio_threshold * n.distance:
                    good_bwd.append(m)

        # --- Step C: Cross-Check Verification ---
        # A match is kept only if Frame1[i] maps to Frame2[j] AND Frame2[j] maps back to Frame1[i]
        cross_checked_matches = []
        bwd_dict = {b.trainIdx: b.queryIdx for b in good_bwd} # Maps F1_idx -> F2_idx
        
        for f in good_fwd:
            if f.queryIdx in bwd_dict and bwd_dict[f.queryIdx] == f.trainIdx:
                cross_checked_matches.append(f)

        return cross_checked_matches

    def verify_geometry_ransac(self, kp1, kp2, matches, pixel_threshold=1.0):
        """
        Applies a 5-point Epipolar RANSAC constraint over the Essential Matrix 
        to isolate true inliers from moving structural objects or degenerate features.
        """
        if len(matches) < 8:
            return [], None, None

        # Extract matching 2D coordinates
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Compute Essential Matrix with high confidence RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=pixel_threshold
        )

        if mask is None:
            return [], None, None

        # Filter out tracking outliers based on the binary mask array
        inlier_matches = [matches[i] for i in range(len(matches)) if mask[i][0] == 1]
        
        # Recover Relative Motion parameters [Rotation (R), Translation (t)]
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)

        return inlier_matches, R, t


# --- Integration Test Stub ---
if __name__ == "__main__":
    # Test setting using ORB mode configuration
    frontend = VisualFrontend(feature_type="ORB", max_features=1200)
    print(f"Frontend engine initialized with: {frontend.feature_type}")
    
    # Create two mock canvas sheets to ensure math operations bind correctly
    img_mock1 = np.zeros((480, 752), dtype=np.uint8)
    img_mock2 = np.zeros((480, 752), dtype=np.uint8)
    
    # Draw simple trackable geometric targets
    cv2.circle(img_mock1, (300, 200), 40, 255, -1)
    cv2.circle(img_mock1, (150, 350), 20, 255, -1)
    cv2.circle(img_mock2, (295, 202), 40, 255, -1) # Slightly shifted
    cv2.circle(img_mock2, (145, 352), 20, 255, -1) 

    kp1, des1 = frontend.extract_features(img_mock1)
    kp2, des2 = frontend.extract_features(img_mock2)
    
    raw_matches = frontend.match_frames(kp1, des1, kp2, des2)
    inliers, R, t = frontend.verify_geometry_ransac(kp1, kp2, raw_matches)
    
    print(f"🏁 Visual Pipeline Execution Verification:")
    print(f"   -> Extracted Features Frame 1: {len(kp1)} | Frame 2: {len(kp2)}")
    print(f"   -> Cross-Checked Ratio Matches: {len(raw_matches)}")
    print(f"   -> Geometrically Verified Inliers: {len(inliers)}")