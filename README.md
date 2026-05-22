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
