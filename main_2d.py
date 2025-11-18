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
        # FIX: redundant with turns to calc these again?
        # Q: Is there a pattern here?
        # if neither left or right are bigger than mid
        # otherwise just go in the direction that is bigger!
        if (mid > left) and (mid > right):
            direction = self.head
        elif left > right:  # right
            direction = self.head - self.spread
        elif right > left:  # right
            direction = self.head + self.spread
        else:
            direction = r.choice([self.head - self.spread, self.head + self.spread])

        new_pos = (
            self.len * math.sin(direction) + self.pos[0],  # FIX: redundant
            self.len * math.cos(direction) + self.pos[1],
        )
        return new_pos, direction

    def draw(self, canvas: np.ndarray):
        if self.pos[0] >= canvas.shape[0]:
            if self.pos[1] >= canvas.shape[1]:
                if self.pos[0] < 0:
                    if self.pos[1] < 0:
                        self.alive = False
                        return
                        # Kills particles at edge of frame
                        # HACK: it feels weird that this is in draw.

        draw_val = 255  # TODO: will be a 0-1 foloat for vdbs eventually
        canvas[int(self.pos[0])][int(self.pos[1])] = draw_val


sx = 400
sy = 400
fps = 24
rt = 6  # runtime in seconds

decay = 0.95

canvas = np.zeros((sy, sx))  # np uses h*w
particles = []
full_circ_rads = 2 * np.pi


def spawn_random():
    for i in range(1600):
        particles.append(
            Particle(
                pos=(r.randrange(1, sy), r.randrange(1, sx)),
                heading=r.uniform(0, full_circ_rads),
            )
        )


def spawn_rect():
    xpad_l = sx / 4
    xpad_h = xpad_l + (sx / 2)
    ypad_l = sy / 4
    ypad_h = ypad_l + (sy / 2)
    print(f"xpad low: {xpad_l} xpad high: {xpad_h}")
    print(f"ypad low: {ypad_l} ypad high: {ypad_h}")
    for x in range(sx):
        if (x > xpad_l) and (x < xpad_h):
            for y in range(sy):
                if (y > ypad_l) and (y < ypad_h):
                    particles.append(
                        Particle(pos=(x, y), heading=r.uniform(0, full_circ_rads))
                    )


spawn_rect()
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
    print(f"frames {i} rendered")

now = re.sub(r"[:-]", "", datetime.now().isoformat(timespec="seconds"))
imageio.mimsave(
    f"./output/physarum_{now}.gif", frames, fps=fps, subtractrectangles=True, loop=0
)
print("done")
