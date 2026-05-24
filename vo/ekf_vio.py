import numpy as np


class EKFVIO:
    def __init__(self):
        self.p = np.zeros(3)
        self.v = np.zeros(3)

        self.P = np.eye(6) * 0.1  # covariance

    def predict(self, acc, dt):

        g = np.array([0, 0, -9.81])

        self.v += (acc + g) * dt
        self.p += self.v * dt

    def update(self, vo_p):

        # measurement residual
        y = vo_p - self.p

        H = np.eye(3, 6)
        H[:, :3] = np.eye(3)

        R = np.eye(3) * 0.5

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ y

        self.p += dx[:3]
        self.v += dx[3:]

        I = np.eye(6)
        self.P = (I - K @ H) @ self.P