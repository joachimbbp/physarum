import matplotlib.pyplot as plt
import numpy as np
import math
import random


class Particle:
    def __init__(self, pos_and_head: tuple[int, int, float]):
        self.pos = (pos_and_head[0], pos_and_head[1])
        self.heading = pos_and_head[2]
        self.spread = 1.570  # approx 90 degrees in radians
        self.length = 2  # the sensor and move distance

    def search(self, angle: float, canvas) -> float:
        # returns x,y, and if there is something there
        x = self.dist * math.sin(angle) + self.pos[0]
        y = self.dist * math.cos(angle) + self.pos[1]
        return canvas[x][y]

    def sense_and_rotate(self, canvas) -> tuple[int, int, float]:
        # sense the four posibilities on the grid
        turns = [self.heading - self.spread, self.heading + self.spread]
        left = self.search(self, turns[0], canvas)
        mid = self.search(self, self.heading, canvas)
        right = self.search(self, turns[1], canvas)
        direction = random.choice(turns)  # new heading

        if ((left >= 0) and (mid == 0) and (right >= 0)):
            direction = self.heading
        # if ((left == 0.0) and (mid >= 0.0) and (right == 0.0)):
        #     direction = random.choice(turns)
        elif ((left >= 0) and (mid >= 0) and (right == 0)):
            direction = turns[0]
        elif ((left == 0) and (mid >= 0) and (right >= 0)):
            direction = turns[1]
        return (self.length * math.cos(direction) + self.pos[0],
                self.length + math.sin(direction) + self.pos[1],
                direction)

    def draw(self, canvas):
        canvas[self.pos[0]][self.pos[1]] = 255


def draw_canvas(particles, canvas, frame_num):
    for i, p in enumerate(particles):
        p.draw(canvas)
    plt.imsave(f'c_{frame_num}.png', canvas)


canvas = np.zeros((100, 100))
# canvas = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
particles = []
full_circ_rads = 2 * np.pi
# spawn
for i in range(100):
    particles.append(Particle((random.randrange(1, 100),
                               random.randrange(1, 100),
                               random.uniform(0, full_circ_rads))))
draw_canvas(particles, canvas, 1)
# time steps
# for i in range(100):
#     for p in particles:
#         #        particles[p] = Particle.__init__(particles[p].sense
