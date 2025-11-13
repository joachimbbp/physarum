import matplotlib.pyplot as plt
import numpy as np
import math
import random as r
import imageio
from datetime import datetime
import re


class Particle:
    def __init__(self, pos: tuple[int, int], heading: float):
        self.pos = pos
        self.head = heading
        self.spread = 0.785398  # approx 45 degrees in radians
        self.len = 1
        self.alive = True
        # NOTE: All particles will just have a single pixel grow/sense length

    def search(self, angle: float, canvas: np.ndarray) -> float:
        x = self.len * math.sin(angle) + self.pos[0]
        y = self.len * math.cos(angle) + self.pos[1]
        # LLM: clamping logic:
        x = max(0, min(canvas.shape[0] - 1, int(x)))
        y = max(0, min(canvas.shape[1] - 1, int(y)))
        return canvas[int(x)][int(y)]

    def sense_and_rotate(self, canvas):
        turns = [self.head - self.spread, self.head + self.spread]
        left = self.search(turns[0], canvas)
        mid = self.search(self.head, canvas)
        right = self.search(turns[1], canvas)
        direction = r.choice(turns)  # new heading

        # LLM: fix found this out:
        # that little < logic in jenson's sketch *did* mean something
        # turns out this all needed to be relational, not just against zeros!
        if (mid > left) and (mid > right):
            direction = self.head

        elif left > right:
            direction = self.head - self.spread
        elif right > left:
            direction = self.head + self.spread
        else:
            direction = r.choice(
                [self.head - self.spread, self.head + self.spread])

        new_pos = (self.len * math.sin(direction) + self.pos[0],
                   self.len * math.cos(direction) + self.pos[1])
        return new_pos, direction

    def draw(self, canvas: np.ndarray):
        if (self.pos[0] >= canvas.shape[0]) or (self.pos[1] >= canvas.shape[1]) or (self.pos[0] < 0) or (self.pos[1] < 0):
            self.alive = False
            return
            # Kills particles at edge of frame
            # HACK: it feels weird that this is in draw.

        draw_val = 255  # TODO: will be a 0-1 foloat for vdbs eventually
        # print("draw val: ", draw_val)
        canvas[int(self.pos[0])][int(self.pos[1])] = draw_val


sx = 400
sy = 400
fps = 24
rt = 10  # runtime in seconds

decay = 0.95

canvas = np.zeros((sy, sx))  # np uses h*w
particles = []
full_circ_rads = 2 * np.pi
# spawn
for i in range(1600):
    particles.append(Particle(pos=(r.randrange(1, sy), r.randrange(1, sx)),
                              heading=r.uniform(0, full_circ_rads)))

frames = []
# time steps
for i in range(int(fps * rt)):
    new_particles = []
    for p in particles:
        if p.alive:
            p, h = p.sense_and_rotate(canvas)  # naming could be better here
            new_particles.append(Particle(p, h))
    particles = new_particles
    canvas *= decay
    for p in particles:
        p.draw(canvas)
    frames.append(canvas.copy())
    print(f'frames {i} rendered')

now = re.sub(r'[:-]', '', datetime.now().isoformat(timespec='seconds'))
imageio.mimsave(f'./output/physarum_{now}.gif',
                frames, fps=fps, subtractrectangles=True, loop=0)
