import numpy as np

class SimpleVIO:

    def __init__(self):

        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.prev_vo = None

        # CORRIDOR SAFE SETTINGS
        self.vo_weight = 0.98
        self.imu_gain = 0.002

    def step(self, vo_meas, acc, dt=0.01):

        if self.prev_vo is None:
            self.prev_vo = vo_meas.copy()
            self.p = vo_meas.copy()
            return self.p

        dvo = vo_meas - self.prev_vo

        # IMU smoothing only
        self.v = 0.99 * self.v + acc * dt * self.imu_gain
        self.v = np.clip(self.v, -1, 1)   # stability fix

        imu_delta = self.v * dt

        vo_pred = self.p + dvo + imu_delta

        self.p = self.vo_weight * vo_pred + (1 - self.vo_weight) * self.p

        self.prev_vo = vo_meas.copy()

        return self.p