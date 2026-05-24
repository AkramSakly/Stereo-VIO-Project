import cv2
import pandas as pd
import os

# Root project path variables
dataset_path = "data/raw/dataset-room2_512_16/mav0"
image_dir = os.path.join(dataset_path, "cam0/data")
imu_file = os.path.join(dataset_path, "imu0/data.csv")

# Global image list for imports
if os.path.exists(image_dir):
    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
else:
    images = []

def check_data():
    if not images:
        print("Error: No images found. Check your data/raw folder.")
        return
    
    imu_data = pd.read_csv(imu_file)
    print(f"Loaded {len(imu_data)} IMU measurements.")
    print(f"Found {len(images)} images in cam0.")

    for img_name in images[:100]: # Preview first 100
        img_path = os.path.join(image_dir, img_name)
        frame = cv2.imread(img_path)
        cv2.imshow("Data Check", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_data()