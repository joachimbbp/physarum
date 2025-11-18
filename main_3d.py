import numpy as np
import math as m
import random as r
from enum import Enum

import sys

nv_path = "/Users/joachimpfefferkorn/repos/neurovolume/neurovolume/src"
if nv_path not in sys.path:  # LLM:
    sys.path.insert(0, nv_path)
# VERSION: https://github.com/joachimbbp/neurovolume/tree/59a576bafa2dd9035a08a6c2be40c206c9d53d55
import neurovolume as nv


def sphere_to_cart(radial, azimuth, polar) -> tuple[float, float, float]:
    # SOURCE: https://mathworld.wolfram.com/SphericalCoordinates.html
    x = radial * m.cos(azimuth) * m.sin(polar)
    y = radial * m.sin(azimuth) * m.sin(polar)
    z = radial * m.sin(azimuth)
    return x, y, z


def scale_vec(vec: tuple[float, float, float], scalar: float) -> tuple[float, float]:
    # there is probably a numpy way to do this
    return (vec[0] * scalar, vec[1] * scalar, vec[2] * scalar)


def weighted_dir(vecs: [tuple[float, float]]) -> tuple[float, float]:
    # again, def a numpy implementation and maybe it's better
    res = (0.0, 0.0)
    for v in vecs:
        res[0] += v[0]
        res[1] += v[1]
    # I've been using [top, bottom],
    # arctan2 uses [bottom, top]
    return np.arctan2(res[1], res[0])


class Qd(Enum):  # Quadrants
    # Each quadrant starting value in radians
    # as we rotate counter-clockwise
    # tuple is input as (x, y, z) or
    # (bottom vector num, topvector num)
    # The motion will always be weighted forward in the "z" direction
    # HACK: there is probably a numpy specific way to do this!
    D = np.arctan2(-1, -1, 1)  # DOWN
    R = np.arctan2(-1, 1, 1)  # RIGHT
    U = np.arctan2(1, 1, 1)  # UP
    L = np.arctan2(1, -1, 1)  # LEFT


class Particle:
    def __init__(self, pos: tuple[int, int, int], heading: tuple[float, float, float]):
        self.pos = pos
        self.head = heading
        self.spread = 0.785398  # approx 45 degrees in radians
        self.len = 1
        self.alive = True
        # NOTE: All particles will just have a single pixel grow/sense length

    def search(self, canvas: np.ndarray, pos: tuple[float, float, float]) -> float:
        # HACK: just casting to int is probably not the most accurate nearest-neighbor
        # Clamp: (probably some clever loop to DRY this)
        x = max(0, min(canvas.shape[0] - 1, int(pos[0])))
        y = max(0, min(canvas.shape[1] - 1, int(pos[1])))
        z = max(0, min(canvas.shape[2] - 1, int(pos[2])))
        return canvas[int(x)][int(y)][int(z)]

    def sense_and_rotate(self, canvas):
        scaled_vecs = []
        # z will always be the heading

        # WARN: full on spaghetti at walls here
        lv = (  # left vector
            self.head[0] - self.spread,
            self.head[1],
            self.head[2],
        )
        ls = self.search(canvas)  # left scalar
        l_scaled = scale_vec(lv, ls)
        scaled_vecs.append(l_scaled)

        rv = (  # right vector
            self.head[0] + self.spread,
            self.head[1],
            self.head[2],
        )
        rs = self.search(rv, canvas)  # right scalar
        r_scaled = scale_vec(rv, rs)
        scaled_vecs.append(r_scaled)

        uv = (  # up vector
            self.head[0],
            self.head[1] + self.spread,
            self.head[2],
        )
        us = self.search(uv, canvas)  # up scalar
        u_scaled = scale_vec(uv, us)
        scaled_vecs.append(u_scaled)

        dv = (  # down vector
            self.head[0],
            self.head[1] - self.spread,
            self.head[2],
        )
        ds = self.search(dv, canvas)  # down scalar
        d_scaled = scale_vec(dv, ds)
        scaled_vecs.append(d_scaled)

        ms = self.search(self.head, canvas)  # mid scalar
        m_scaled = scale_vec(self.head, ms)
        scaled_vecs.append(m_scaled)

        wd = weighted_dir(scaled_vecs)  # weighted dir

        new_pos = (-1, 0)
        if (wd >= Qd.D.value) and (wd < Qd.R.value):
            new_pos = (-1, 0)
        elif (wd >= Qd.R.value) and (wd < Qd.U.value):
            new_pos = (1, 0)
        elif (wd >= Qd.U.value) and (wd < Qd.L.value):
            new_pos = (0, 1)
        elif (wd >= Qd.L.value) and (wd < Qd.D.value):
            new_pos = (-1, 0)
        else:
            raise ValueError(f"undefined new_pos error. wd = {wd}")
        new_heading = np.arctan2(new_pos[1], new_pos[0])
        return new_pos, new_heading

    def draw(self, canvas: np.ndarray):
        if self.pos[0] >= canvas.shape[0]:
            if self.pos[1] >= canvas.shape[1]:
                if self.pos[2] >= canvas.shape[2]:
                    if (self.pos[0] < 0) or (self.pos[1] < 0):
                        self.alive = False
                        return
            # Kills particles at edge of frame
            # HACK: it feels weird that this is in draw.

        draw_val = 1.0
        canvas[int(self.pos[0])][int(self.pos[1])[int(self.pos[2])]] = draw_val


sx = 100
sy = 100
sz = 100
fps = 24
rt = 6  # runtime in seconds

decay = 0.95

canvas = np.zeros((sy, sx, sz), dtype=np.float64)  # np uses h*w
particles = []

full_circ_rads = 2 * np.pi

print("random spawning")


def spawn_random():
    for i in range(500):
        particles.append(
            Particle(
                pos=(r.randrange(1, sy), r.randrange(1, sx), r.randrange(1, sx)),
                heading=r.uniform(0, full_circ_rads),
            )
        )


print("spawned")

spawn_random()
print(len(particles))
for p in particles:
    print(type(p))
frames = []
# # time steps
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

# affine_identity = np.array([ # LLM:
#     [1.0, 0.0, 0.0, 0.0],  # x-axis
#     [0.0, 1.0, 0.0, 0.0],  # y-axis
#     [0.0, 0.0, 1.0, 0.0],  # z-axis
#     [0.0, 0.0, 0.0, 1.0],  # homogeneous coord
# ], dtype=np.float64)

# for i, f in enumerate(frames):
#     o = f'./output/physarum_{i}.vdb'
#     nv.ndarray_to_VDB(f, o, affine_identity)
# print('done')
