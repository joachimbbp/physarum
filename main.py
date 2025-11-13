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
        self.age = 0
        self.health = 5
        # NOTE: All particles will just have a single pixel grow/sense length

    def decay(self, decay='linear', decay_factor=1):
        self.age += 1
        match decay:
            case 'linear':
                self.health = self.health - (self.age * decay_factor)
            case 'exponential':  # WARN: untested
                self.health = int(self.health / (self.age * decay_factor))

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

        if ((left >= 0) and (mid == 0) and (right >= 0)):
            direction = self.head
        # if ((left == 0.0) and (mid >= 0.0) and (right == 0.0)):
        #     direction = random.choice(turns)
        elif ((left >= 0) and (mid >= 0) and (right == 0)):
            direction = turns[0]
        elif ((left == 0) and (mid >= 0) and (right >= 0)):
            direction = turns[1]
        new_pos = (self.len * math.sin(direction) + self.pos[0],
                   self.len + math.cos(direction) + self.pos[1])
        return new_pos, direction

    def draw(self, canvas: np.ndarray):
        if (self.pos[0] >= canvas.shape[0]) or (self.pos[1] >= canvas.shape[1]) or (self.pos[0] < 0) or (self.pos[1] < 0):
            self.health = 0
            return
            # Kills particles at edge of frame
            # HACK: it feels weird that this is in draw.
        canvas[int(self.pos[0])][int(self.pos[1])] = 255
        # TODO: probably better as a 0-1 float tbh


canvas = np.zeros((100, 100))
# canvas = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
particles = []
full_circ_rads = 2 * np.pi
# spawn
for i in range(100):
    particles.append(Particle(pos=(r.randrange(1, 100), r.randrange(1, 100)),
                              heading=r.uniform(0, full_circ_rads)))

frames = []
fps = 24
# time steps
for i in range(fps * 10):
    new_particles = []
    for p in particles:
        if p.health > 0:
            p, h = p.sense_and_rotate(canvas)  # naming could be better here
            new_particles.append(Particle(p, h))
    particles = new_particles
    for p in particles:
        p.decay()
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
