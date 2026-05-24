import numpy as np
from optimizer import SlidingWindow


class VIO_SW:
    def __init__(self, window_size=15):
        self.window = SlidingWindow(window_size)
        self.p = np.zeros(3)

    def step(self, vo_p, imu_p):

        # simple fusion (stable learning version)
        fused = 0.7 * vo_p + 0.3 * imu_p

        self.window.add(vo_p, imu_p)

        optimized = self.window.optimize()

        self.p = optimized[-1]

        return self.p