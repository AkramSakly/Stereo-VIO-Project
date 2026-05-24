import numpy as np

class IMUPreintegration:
    def __init__(self):
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.R = np.eye(3)

        self.bg = np.zeros(3)
        self.ba = np.zeros(3)

        self.g = np.array([0, 0, -9.81])

    def reset(self):
        self.p[:] = 0
        self.v[:] = 0
        self.R[:] = np.eye(3)

    def skew(self, w):
        return np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ])

    def integrate(self, acc, gyro, dt):

        gyro = gyro - self.bg
        acc = acc - self.ba

        omega = gyro * dt
        theta = np.linalg.norm(omega)

        if theta > 1e-8:
            axis = omega / theta
            K = self.skew(axis)

            dR = (
                np.eye(3)
                + np.sin(theta) * K
                + (1 - np.cos(theta)) * (K @ K)
            )
        else:
            dR = np.eye(3)

        self.R = self.R @ dR

        acc_world = self.R @ acc + self.g

        self.v += acc_world * dt
        self.p += self.v * dt + 0.5 * acc_world * dt * dt

        return self.p.copy()