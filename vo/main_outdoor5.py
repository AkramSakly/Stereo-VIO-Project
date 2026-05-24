import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from loader import load_images, load_imu, load_groundtruth
from vo import MonocularVO
from vio import SimpleVIO
from evaluate import compute_ate, compute_rpe, align_umeyama


# =========================
# DATASET PATH (CHANGE ONLY THIS)
# =========================

DATASET = r"C:\Users\selva\vio_project\data\raw\dataset-outdoors5_512_16\mav0"

CAM_DIR = os.path.join(DATASET, "cam0", "data")
IMU_CSV = os.path.join(DATASET, "imu0", "data.csv")
GT_CSV  = os.path.join(DATASET, "mocap0", "data.csv")


# =========================
# CHECK PATH
# =========================

print("CAM_DIR:", CAM_DIR)
print("EXISTS:", os.path.exists(CAM_DIR))


# =========================
# LOAD DATA
# =========================

images = load_images(CAM_DIR)
imu_df = load_imu(IMU_CSV)
gt = load_groundtruth(GT_CSV)


# =========================
# CAMERA MATRIX
# =========================

K = np.array([
    [190.978, 0, 254.931],
    [0, 190.973, 256.897],
    [0, 0, 1]
])


# =========================
# SYSTEMS
# =========================

vo = MonocularVO(K)
vio = SimpleVIO()


# =========================
# TRAJECTORIES
# =========================

vo_traj = []
vio_traj = []

imu_i = 0

print("\n[INFO] Running VO + VIO pipeline...")


# =========================
# MAIN LOOP
# =========================

for i in tqdm(range(len(images) - 1)):

    # =====================
    # VO
    # =====================
    vo.process(images[i], images[i + 1])

    vo_p = vo.pose[:3, 3].copy()

    # safety
    if np.any(np.isnan(vo_p)):
        vo_p = np.zeros(3)

    vo_traj.append(vo_p)

    # =====================
    # IMU
    # =====================
    if imu_i < len(imu_df) - 2:
        acc = np.array([
            imu_df.iloc[imu_i]['a_RS_S_x [m s^-2]'],
            imu_df.iloc[imu_i]['a_RS_S_y [m s^-2]'],
            imu_df.iloc[imu_i]['a_RS_S_z [m s^-2]']
        ])
        imu_i += 1
    else:
        acc = np.zeros(3)

    # =====================
    # VIO
    # =====================
    vio_p = vio.step(vo_p, acc, dt=0.01)

    if np.any(np.isnan(vio_p)):
        vio_p = np.zeros(3)

    vio_traj.append(vio_p)


# =========================
# CONVERT
# =========================

vo_traj = np.array(vo_traj)
vio_traj = np.array(vio_traj)
gt = np.array(gt)


# =========================
# ALIGN LENGTH
# =========================

N = min(len(vo_traj), len(vio_traj))

vo_traj = vo_traj[:N]
vio_traj = vio_traj[:N]

idx = np.linspace(0, len(gt) - 1, N).astype(int)
gt = gt[idx]


# =========================
# ALIGN TO GT
# =========================

vo_aligned = align_umeyama(vo_traj, gt)
vio_aligned = align_umeyama(vio_traj, gt)


# =========================
# METRICS
# =========================

ate_vo = compute_ate(gt, vo_aligned)
ate_vio = compute_ate(gt, vio_aligned)

rpe_vo = compute_rpe(gt, vo_aligned)
rpe_vio = compute_rpe(gt, vio_aligned)


print("\n===== FINAL RESULTS =====")
print("VO  ATE:", ate_vo)
print("VIO ATE:", ate_vio)
print("VO  RPE:", rpe_vo)
print("VIO RPE:", rpe_vio)


# =========================
# SAVE RESULTS
# =========================

os.makedirs("results", exist_ok=True)

with open("results/final_results.txt", "w") as f:
    f.write("FINAL RESULTS\n")
    f.write(f"VO ATE: {ate_vo}\n")
    f.write(f"VIO ATE: {ate_vio}\n")
    f.write(f"VO RPE: {rpe_vo}\n")
    f.write(f"VIO RPE: {rpe_vio}\n")


# =========================
# PLOT TRAJECTORY
# =========================

plt.figure(figsize=(10, 8))

plt.plot(gt[:, 0], gt[:, 1], 'g-', linewidth=3, label="GT")
plt.plot(vo_aligned[:, 0], vo_aligned[:, 1], 'r--', label="VO")
plt.plot(vio_aligned[:, 0], vio_aligned[:, 1], 'b-', label="VIO")

plt.legend()
plt.axis("equal")
plt.grid()
plt.title("Trajectory Comparison")

plt.savefig("results/trajectory.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================
# DRIFT PLOT
# =========================

drift_vo = np.linalg.norm(gt - vo_aligned, axis=1)
drift_vio = np.linalg.norm(gt - vio_aligned, axis=1)

plt.figure(figsize=(10, 5))

plt.plot(drift_vo, 'r', label="VO Drift")
plt.plot(drift_vio, 'b', label="VIO Drift")

plt.legend()
plt.grid()
plt.title("Drift Comparison")

plt.savefig("results/drift.png", dpi=300, bbox_inches="tight")
plt.show()