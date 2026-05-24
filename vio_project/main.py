from vo import VisualOdometry
from vio import VIOSystem
from loader import DataLoader

def main():
    data = DataLoader('dataset-room2_512_16/mav0')
    vio = VIOSystem()

    for frame in data:
        vio.step(frame)

if __name__ == "__main__":
    main()
