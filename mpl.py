import matplotlib.pyplot as plt
import numpy as np
import math
import random

canvas = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

plt.imsave("where.png", canvas)
