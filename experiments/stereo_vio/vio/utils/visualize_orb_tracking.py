import cv2
import os
import glob
import numpy as np

class ORBFrontendVisualizer:
    def __init__(self, base_project_path=r"C:\Users\selva\vio_project"):
        self.base_path = base_project_path
        self.output_dir = os.path.join(self.base_path, "exported_media", "orb_tracking")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 Created ORB media output directory at: {self.output_dir}")

        # Camera Intrinsics Matrix (Cam0)
        self.K = np.array([
            [190.97,   0.0,  254.93],
            [  0.0,  190.68, 252.50],
            [  0.0,    0.0,    1.0]
        ], dtype=np.float32)

        # Frontend Parameters
        self.max_features = 1200
        self.ransac_thresh = 1.0

        # Initialize ORB Detector and Matcher
        self.orb = cv2.ORB_create(nfeatures=self.max_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def find_camera_folder(self, dataset_name):
        """Locates the absolute path where raw image frames reside."""
        sequence_path = os.path.join(self.base_path, "data", "raw", dataset_name)
        search_pattern = os.path.join(sequence_path, "**/cam0/data")
        found_folders = glob.glob(search_pattern, recursive=True)
        return found_folders[0] if found_folders else None

    def filter_outliers_ransac(self, kp1, kp2, matches):
        """Applies Epipolar RANSAC over the Essential Matrix to isolate inliers."""
        if len(matches) < 8:
            return []
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=self.ransac_thresh)
        if mask is None:
            return []
        return [matches[i] for i in range(len(matches)) if mask[i][0] == 1]

    def process_and_export(self, dataset_name, max_frames=800):
        img_dir = self.find_camera_folder(dataset_name)
        if not img_dir or not os.path.exists(img_dir):
            print(f"❌ Target path missing for: {dataset_name}")
            return

        all_frames = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))])
        total_frames = len(all_frames)
        if total_frames == 0:
            return

        # Setup Video Writer dimensions based on a side-by-side match image
        sample_img = cv2.imread(os.path.join(img_dir, all_frames[0]), 0)
        h, w = sample_img.shape
        # Side-by-side drawing width will be 2 * w
        video_dimensions = (w * 2, h)

        output_video_path = os.path.join(self.output_dir, f"{dataset_name}_orb_tracking.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 20, video_dimensions)

        print(f"\n🎬 Extracting and Visualizing ORB Features for: {dataset_name}")
        
        # Load Frame 0
        prev_img = cv2.imread(os.path.join(img_dir, all_frames[0]), 0)
        prev_kp, prev_des = self.orb.detectAndCompute(prev_img, None)

        export_limit = min(total_frames, max_frames)
        sample_saved = False

        for idx in range(1, export_limit):
            curr_img = cv2.imread(os.path.join(img_dir, all_frames[idx]), 0)
            kp, des = self.orb.detectAndCompute(curr_img, None)

            if des is not None and prev_des is not None:
                # 1. Compute raw descriptor matches
                raw_matches = self.bf.match(prev_des, des)
                # 2. Filter via Geometric RANSAC
                valid_matches = self.filter_outliers_ransac(prev_kp, kp, raw_matches)

                # 3. Draw tracking matches side-by-side
                # Green lines/dots represent valid, filtered inlier tracking points
                display_frame = cv2.drawMatches(
                    prev_img, prev_kp, curr_img, kp, valid_matches[:50], None,
                    matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

                # 4. Burn status text overlay into the frame
                cv2.putText(display_frame, f"Dataset: {dataset_name}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, f"Tracked Inliers: {len(valid_matches)} / {len(raw_matches)}", (30, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Convert grayscale output canvas to standard BNC 3-channel for video format
                if len(display_frame.shape) == 2:
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

                video_writer.write(display_frame)

                # Save an exact high-resolution standalone photo for your slides/report text
                if idx == 100 and not sample_saved:
                    photo_path = os.path.join(self.output_dir, f"sample_snapshot_{dataset_name}.png")
                    cv2.imwrite(photo_path, display_frame)
                    print(f"   📸 Saved standalone snapshot to: {os.path.basename(photo_path)}")
                    sample_saved = True

            # Prepare for next iteration loop
            prev_img = curr_img.copy()
            prev_kp, prev_des = kp, des

            if idx % 200 == 0 or idx == export_limit - 1:
                print(f"   Processed ORB frame association {idx}/{export_limit}...")

        video_writer.release()
        print(f"💾 Saved complete tracked feature visualization video to:\n   👉 {output_video_path}")

if __name__ == "__main__":
    visualizer = ORBFrontendVisualizer()
    datasets = ["dataset-room2_512_16", "dataset-corridor3_512_16", "dataset-outdoors5_512_16"]
    
    for ds in datasets:
        visualizer.process_and_export(ds, max_frames=800)