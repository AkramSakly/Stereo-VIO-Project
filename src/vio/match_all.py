import cv2
import numpy as np
import os
import pandas as pd

class FeatureMatcher:
    def __init__(self):
        # Initializing ORB with 1200 features as per your VIO project specs
        self.orb = cv2.ORB_create(nfeatures=1200)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def process_sequence(self, dataset_name):
        base_path = f"data/raw/{dataset_name}/mav0/cam0/data"
        if not os.path.exists(base_path):
            print(f"❌ Path not found: {base_path}")
            return

        img_list = sorted(os.listdir(base_path))
        print(f"🔍 Matching {len(img_list)} frames for: {dataset_name}")

        prev_img = cv2.imread(os.path.join(base_path, img_list[0]), 0)
        prev_kp, prev_des = self.orb.detectAndCompute(prev_img, None)

        match_counts = []

        for i in range(1, len(img_list)):
            curr_img = cv2.imread(os.path.join(base_path, img_list[i]), 0)
            kp, des = self.orb.detectAndCompute(curr_img, None)

            if des is not None and prev_des is not None:
                matches = self.matcher.match(prev_des, des)
                match_counts.append(len(matches))
                
                # Update for next iteration
                prev_kp, prev_des = kp, des
            
            if i % 1000 == 0:
                print(f"Processed {i} frames...")

        avg_matches = np.mean(match_counts)
        print(f"✅ {dataset_name} Done. Avg Matches: {avg_matches:.2f}\n")
        return avg_matches

if __name__ == "__main__":
    datasets = [
        "dataset-room2_512_16",
        "dataset-corridor3_512_16",
        "dataset-outdoors5_512_16"
    ]
    
    fm = FeatureMatcher()
    results = {}
    
    for ds in datasets:
        results[ds] = fm.process_sequence(ds)

    # Summary Table for your Week 11 Report
    print("--- Final Matching Summary ---")
    for ds, count in results.items():
        print(f"{ds}: {count} avg matches/frame")