import numpy as np

class VINSVIO:
    def __init__(self):

        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.P = np.eye(6) * 0.1

    def predict(self, imu_p):

        # IMU gives motion prior
        self.p = 0.8 * self.p + 0.2 * imu_p

    def update(self, vo_p):

        H = np.zeros((3,6))
        H[:3,:3] = np.eye(3)

        R = np.eye(3) * 0.02

        y = vo_p - self.p

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        x = np.hstack([self.p, self.v])
        x = x + K @ y

        self.p = x[:3]
        self.v = x[3:]

        self.P = (np.eye(6) - K @ H) @ self.P

    def step(self, vo_p, imu_p):

        self.predict(imu_p)
        self.update(vo_p)

        return self.p