import numpy as np
import math as m
import random as r
import imageio
from datetime import datetime
import re


def sphere_to_cart(radial, azimuth, polar) -> tuple[float, float, float]:
    # SOURCE: https://mathworld.wolfram.com/SphericalCoordinates.html
    x = radial * m.cos(azimuth) * m.sin(polar)
    y = radial * m.sin(azimuth) * m.sin(polar)
    z = radial * m.sin(azimuth)
    return x, y, z


class Particle:
    def __init__(self, pos: tuple[int, int, int],
                 heading: tuple[float, float]):  # yaw and pitch
        self.pos = pos
        self.head = heading
        self.spread = 0.785398  # approx 45 degrees in radians
        self.len = 1
        self.alive = True
        # NOTE: All particles will just have a single pixel grow/sense length

    def search(self, xr: float, yr: float, canvas: np.ndarray) -> float:
        # VERIFY: but I think:
        # self.len, the growth/search direction is the radial
        # xr is the azimuth (x direction)
        # yr is the polar (y direction)
        x, y, z = sphere_to_cart(self.len, xr, yr)

        # Clamp: (probably some clever loop to DRY this)
        x = max(0, min(canvas.shape[0] - 1, int(x)))
        y = max(0, min(canvas.shape[1] - 1, int(y)))
        z = max(0, min(canvas.shape[2] - 1, int(z)))
        return canvas[int(x)][int(y)][int(z)]

    def sense_and_rotate(self, canvas):
        towards = self.head - self.spread  # right and up
        away = self.head + self.spread  # left and down

        left = self.search(away, 0, canvas)
        right = self.search(towards, 0, canvas)
        up = self.search(0, towards, canvas)
        down = self.search(0, away, canvas)
        mid = self.search(self.head[0], self.head[1], canvas)

        non_mid = [left, right, up, down]
        straight_ahead = True
        for d in non_mid:
            if mid < d:
                straight_ahead = False
        if straight_ahead:
            direction = self.head

        else:
            # Here's where this gets hard
            # we have to weight the direction!

    def draw(self, canvas: np.ndarray):
        if (self.pos[0] >= canvas.shape[0]) or (self.pos[1] >= canvas.shape[1]) or (self.pos[0] < 0) or (self.pos[1] < 0):
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
        particles.append(Particle(pos=(r.randrange(1, sy), r.randrange(1, sx)),
                                  heading=r.uniform(0, full_circ_rads)))


def spawn_rect():
    xpad_l = sx / 4
    xpad_h = xpad_l + (sx/2)
    ypad_l = sy / 4
    ypad_h = ypad_l + (sy/2)
    print(f'xpad low: {xpad_l} xpad high: {xpad_h}')
    print(f'ypad low: {ypad_l} ypad high: {ypad_h}')
    for x in range(sx):
        if (x > xpad_l) and (x < xpad_h):
            for y in range(sy):
                if (y > ypad_l) and (y < ypad_h):
                    particles.append(
                        Particle(pos=(x, y),
                                 heading=r.uniform(0, full_circ_rads)))


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
    print(f'frames {i} rendered')

now = re.sub(r'[:-]', '', datetime.now().isoformat(timespec='seconds'))
imageio.mimsave(f'./output/physarum_{now}.gif',
                frames, fps=fps, subtractrectangles=True, loop=0)
print('done')
