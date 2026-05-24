import numpy as np

class IMU:

    def __init__(self):

        self.p = np.zeros(3)
        self.v = np.zeros(3)

    def step(self, acc, dt):

        g = np.array([0, 0, -9.81])

        self.v += (acc + g) * dt
        self.p += self.v * dt

        return self.p