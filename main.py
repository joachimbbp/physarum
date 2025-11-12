import matplotlib.pyplot as plt
import numpy as np
import math
import random
import imageio
from datetime import datetime
import re


class Particle:
    def __init__(self, pos_and_head: tuple[int, int, float]):
        self.pos = (pos_and_head[0], pos_and_head[1])
        self.heading = pos_and_head[2]
        self.spread = 1.570  # approx 90 degrees in radians
        self.length = 2  # the sensor and move distance

    def search(self, angle: float, canvas: np.ndarray) -> float:
        x = self.length * math.sin(angle) + self.pos[0]
        y = self.length * math.cos(angle) + self.pos[1]
        # TODO: Flip heading. For now we'll just clamp it
        # LLM: clamping logic
        x = max(0, min(canvas.shape[0] - 1, int(x)))
        y = max(0, min(canvas.shape[1] - 1, int(y)))
        return canvas[int(x)][int(y)]

    def sense_and_rotate(self, canvas) -> tuple[int, int, float]:
        turns = [self.heading - self.spread, self.heading + self.spread]
        left = self.search(turns[0], canvas)
        mid = self.search(self.heading, canvas)
        right = self.search(turns[1], canvas)
        direction = random.choice(turns)  # new heading

        if ((left >= 0) and (mid == 0) and (right >= 0)):
            direction = self.heading
        # if ((left == 0.0) and (mid >= 0.0) and (right == 0.0)):
        #     direction = random.choice(turns)
        elif ((left >= 0) and (mid >= 0) and (right == 0)):
            direction = turns[0]
        elif ((left == 0) and (mid >= 0) and (right >= 0)):
            direction = turns[1]
        return (self.length * math.sin(direction) + self.pos[0],
                self.length + math.cos(direction) + self.pos[1],
                direction)

    def draw(self, canvas: np.ndarray):
        # WARN: clamping doesn't work above
        if (self.pos[0] >= canvas.shape[0]) or (self.pos[1] >= canvas.shape[1]):
            return
        canvas[int(self.pos[0])][int(self.pos[1])] = 255
        # TODO: probably better as a 0-1 float tbh


def draw_canvas(particles, canvas, frame_num, name="physarum"):
    for i, p in enumerate(particles):
        p.draw(canvas)
    plt.imsave(f'./output/{name}_{frame_num}.png', canvas)


canvas = np.zeros((100, 100))
# canvas = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
particles = []
full_circ_rads = 2 * np.pi
# spawn
for i in range(100):
    particles.append(Particle((random.randrange(1, 100),
                               random.randrange(1, 100),
                               random.uniform(0, full_circ_rads))))

frames = []
fps = 24
# time steps
for i in range(fps * 10):
    new_particles = []
    for p in particles:
        new_vec = p.sense_and_rotate(canvas)
        new_particles.append(Particle(new_vec))
    particles = new_particles
    for p in particles:
        p.draw(canvas)
    frames.append(canvas.copy())
    print(f'frames {i} rendered')

now = re.sub(r'[:-]', '', datetime.now().isoformat(timespec='seconds'))
imageio.mimsave(f'./output/physarum_{now}.gif', frames, fps=fps)
# draw_canvas(particles, canvas, i)
# print(f'frame {i} drawn')
#
# ffmpeg -framerate 10 -i physarum_%d.png output.gif

# create
