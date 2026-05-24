class VIOSystem:
    def __init__(self):
        self.vo = None
        self.imu = None
        self.state = None

    def step(self, frame):
        # frame = (img, imu)
        img, imu = frame if isinstance(frame, tuple) else (frame, None)
        return None
