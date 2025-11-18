import math
import scipy
from enum import Enum
import numpy as np
# Just testing


class WaterVapor(float, Enum):
    VERY_DRY = np.degrees(np.arctan2(1, 1))
    DRY = 0.5
    MEDIAN = 0.8
    WET = 1.0


class Quadrants(Enum):
    # Each quadrant starting value in radians
    # as we rotate counter-clockwise
    # tuple is input as (y, x) or
    ZTEST = np.arctan2(0, 1)
    DOWN = np.arctan2(-1, -1)
    RIGHT = np.arctan2(-1, 1)
    UP = np.arctan2(1, 1)
    LEFT = np.arctan2(1, -1)


for q in Quadrants:
    print(q)
    print(q.value)
    print(np.degrees(q.value))
    print("\n")
