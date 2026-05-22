# Stereo-VIO-Pr 🚀  
Visual-Inertial Odometry Pipeline on TUM-VI Benchmark

---

## 📌 Project Overview

This project implements a **Visual Odometry (VO) and Visual-Inertial Odometry (VIO)** pipeline for camera pose estimation using the **TUM-VI dataset**.

The system estimates the camera trajectory using:
- ORB feature extraction and matching
- Essential matrix estimation (RANSAC)
- IMU-based motion smoothing
- Trajectory evaluation using ATE and RPE metrics

The pipeline is evaluated on:
- Room2
- Corridor3
- Outdoor5

---

## 🧠 Key Features

- Monocular Visual Odometry (VO)
- Lightweight IMU integration (VIO)
- Trajectory estimation in metric scale (aligned)
- Evaluation using:
  - Absolute Trajectory Error (ATE)
  - Relative Pose Error (RPE)
- Visualization of trajectories and drift

---

## 📂 Project Structure
Stereo-VIO-Pr/
│
├── vo/
│ ├── main.py
│ ├── vo.py
│ ├── vio.py
│ ├── loader.py
│ ├── evaluate.py
│
├── data/
│ ├── raw/
│ ├── dataset-room2/
│ ├── dataset-corridor3/
│ ├── dataset-outdoor5/
│
├── results/
│ ├── trajectory.png
│ ├── drift.png
│
├── README.md
└── requirements.txt




 Installation

```bash
git clone https://github.com/your-username/Stereo-VIO-Pr.git
cd Stereo-VIO-Pr
pip install -r requirements.txt

📊 Results
Room2
VO ATE: ~1.247

VIO ATE: ~1.245

Corridor3
VO ATE: ~1.00

VIO ATE: ~1.00

Outdoor5
VO ATE: ~0.65

VIO ATE: ~0.65

