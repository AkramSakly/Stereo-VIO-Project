import cv2
import os
import glob

class DatasetVideoExporter:
    def __init__(self, base_project_path=r"C:\Users\selva\vio_project"):
        self.base_path = base_project_path
        self.output_dir = os.path.join(self.base_path, "exported_media")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 Created media output directory at: {self.output_dir}")

    def find_camera_folder(self, dataset_name):
        """Locates the absolute directory path where raw png files reside."""
        sequence_path = os.path.join(self.base_path, "data", "raw", dataset_name)
        search_pattern = os.path.join(sequence_path, "**/cam0/data")
        found_folders = glob.glob(search_pattern, recursive=True)
        
        if found_folders:
            return found_folders[0]
        return None

    def export_video(self, dataset_name, fps=30, max_frames_to_export=1200):
        """
        Reads raw frame images sequentially from cam0 and compiles them 
        into a compressed .mp4 video file.
        """
        img_dir = self.find_camera_folder(dataset_name)
        
        if not img_dir or not os.path.exists(img_dir):
            print(f"❌ Target path missing for {dataset_name}. Ensure images are unzipped.")
            return False

        # Retrieve and sort all frame names chronologically by timestamp
        all_frames = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        total_frames = len(all_frames)
        
        if total_frames == 0:
            print(f"⚠️ No frames discovered inside: {img_dir}")
            return False

        # Read the first image to automatically detect height and width dimensions
        sample_img_path = os.path.join(img_dir, all_frames[0])
        sample_img = cv2.imread(sample_img_path)
        height, width, layers = sample_img.shape
        video_dimensions = (width, height)

        # Initialize the VideoWriter stream (Using MP4V codec for high cross-compatibility)
        output_video_path = os.path.join(self.output_dir, f"{dataset_name}_feed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, video_dimensions)

        # Cap the frame length so video files don't grow too massive for slideshows
        export_limit = min(total_frames, max_frames_to_export)
        print(f"\n🎬 Exporting video clip for: {dataset_name}")
        print(f"   -> Image Directory: {os.path.relpath(img_dir, self.base_path)}")
        print(f"   -> Processing {export_limit} out of {total_frames} total frames...")

        for i in range(export_limit):
            img_path = os.path.join(img_dir, all_frames[i])
            frame = cv2.imread(img_path)
            
            # Optional structural visualization overlays for defense:
            # We can stamp the sequence name and current frame tracking count index onto the video frame
            cv2.putText(frame, f"Seq: {dataset_name}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Frame: {i}/{total_frames}", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            
            video_writer.write(frame)

            if (i + 1) % 400 == 0 or (i + 1) == export_limit:
                print(f"   Compiled frame {i + 1}/{export_limit}...")

        # Release video file resource handle cleanly
        video_writer.release()
        print(f"💾 Video successfully compiled and saved to:\n   👉 {output_video_path}\n")
        return True


if __name__ == "__main__":
    exporter = DatasetVideoExporter()
    
    target_datasets = [
        "dataset-room2_512_16",
        "dataset-corridor3_512_16",
        "dataset-outdoors5_512_16"
    ]
    
    for ds in target_datasets:
        # Exporting up to 1200 frames gives you a perfect ~40 second clip at 30 FPS for presentations
        exporter.export_video(ds, fps=30, max_frames_to_export=1200)