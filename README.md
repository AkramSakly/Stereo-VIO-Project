# Monocular Visual Odometry with Lightweight IMU Fusion (TUM-VI)

## 📌 Overview
This project implements a monocular visual odometry (VO) pipeline enhanced with lightweight IMU fusion.  
The system estimates camera motion using ORB features and classical geometric methods, and is evaluated on the TUM-VI benchmark dataset.

---

## 📌 Features
- Monocular visual odometry (ORB-based)
- Feature matching with Hamming distance
- Outlier rejection using Lowe’s ratio test + RANSAC
- Essential matrix-based motion estimation
- Lightweight IMU smoothing
- Evaluation on TUM-VI sequences (Room2, Corridor3, Outdoor5)

---

## 📂 Project Structure│
├── README.md
└── requirements.txt
vo_project/
│── vo/
│ │── main.py
│ │── loader.py
│ │── imu.py
│ │── utils.py
│── data/
│── results/
│── requirements.txt
│── README.md


---

## ⚙️ Installation

```bash
pip install -r requirements.txt

python main_room2.py
python main_corridor3.py
python main_outdoor5.py



 Installation

```bash
git clone https://github.com/AkramSakly
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

