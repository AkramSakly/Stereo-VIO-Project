import numpy as np
from scipy.optimize import least_squares


class VINSOptimizer:
    def optimize(self, vo, imu):

        vo = np.array(vo)
        imu = np.array(imu)

        N = min(len(vo), len(imu))
        vo = vo[:N]
        imu = imu[:N]

        x0 = vo.flatten()

        def residuals(x):
            x = x.reshape(-1, 3)

            r_vo = x - vo
            r_imu = x - imu

            return np.hstack([
                0.7 * r_vo.flatten(),
                0.3 * r_imu.flatten()
            ])

        result = least_squares(
            residuals,
            x0,
            loss="huber",
            max_nfev=80
        )

        return result.x.reshape(-1, 3)