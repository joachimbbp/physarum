import matplotlib.pyplot as plt
import numpy as np

canvas = np.zeros((100, 100))
# canvas = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


class Particle:
    def __init__(self,
                 pos: tuple[int, int, int],
                 spread: float, distance: int, rotation: float):
        self.pos = pos
        self.spread = spread  # the spread of the three sensor probes in radians
        self.distance = distance  # the sensor and move distance
        self.rotation = rotation  # rotation of entire sensor in radians

    def sense_and_rotate(self, environment) -> tuple[int, int, int]:
        # sense the four posibilities on the grid
        pass


plt.imshow(canvas)
plt.axis('off')
plt.show()
