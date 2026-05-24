import numpy as np
from scipy.optimize import least_squares


class SlidingWindowVIO:
    def __init__(self, window_size=20):
        self.w = window_size

    def optimize(self, vo_traj, imu_traj):

        vo = np.array(vo_traj)
        imu = np.array(imu_traj)

        N = min(len(vo), len(imu))
        vo = vo[:N]
        imu = imu[:N]

        # only optimize last window (IMPORTANT VINS concept)
        start = max(0, N - self.w)

        vo_w = vo[start:N]
        imu_w = imu[start:N]

        x0 = vo_w.flatten()

        def residuals(x):
            x = x.reshape(-1, 3)

            r_vo = x - vo_w
            r_imu = x - imu_w

            # VO strong constraint, IMU weak constraint
            return np.hstack([
                0.8 * r_vo.flatten(),
                0.2 * r_imu.flatten()
            ])

        result = least_squares(
            residuals,
            x0,
            loss="huber",
            max_nfev=100
        )

        optimized_window = result.x.reshape(-1, 3)

        # rebuild full trajectory
        full = vo.copy()
        full[start:N] = optimized_window

        return full