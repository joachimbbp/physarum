import numpy as np
import math as m
import random as r


def sphere_to_cart(radial, azimuth, polar) -> tuple[float, float, float]:
    # SOURCE: https://mathworld.wolfram.com/SphericalCoordinates.html
    x = radial * m.cos(azimuth) * m.sin(polar)
    y = radial * m.sin(azimuth) * m.sin(polar)
    z = radial * m.cos(polar)  # LLM: Suggestion
    return x, y, z


def scale_vec(vec: tuple[float, float, float], scalar: float) -> tuple[float, float]:
    # there is probably a numpy way to do this
    return (vec[0] * scalar, vec[1] * scalar, vec[2] * scalar)


def weighted_dir(vecs: [tuple[float, float]]) -> tuple[float, float, float]:
    # again, def a numpy implementation and maybe it's better
    r0 = 0.0
    r1 = 0.0
    r2 = 0.0
    for v in vecs:
        r0 += v[0]
        r1 += v[1]
        r2 += v[2]
    # WARN: I'm guessing that arctan is backwards like arctan2
    return np.arctan([r2, r1, r0])


# -
class Quadrants:
    # SOURCE: for conversions:
    # https://mathworld.wolfram.com/SphericalCoordinates.html?utm_source=chatgpt.com
    def radial_from_cart(self, x, y, z):
        return np.sqrt(np.square(x) + np.square(y) + np.square(z))

    def azimuth_from_cart(self, x, y):
        return np.arctan2(y, x)  # LLM:

    def polar_from_cart(self, z, r):
        return np.arccos(z / r)  # LLM: fix

    def build_coords(self, x, y, z):
        """'
        returns radial, azimuth, polar as per wiki's convention
        """
        r = self.radial_from_cart(x, y, z)
        azimuth = self.azimuth_from_cart(x, y)
        polar = self.polar_from_cart(z, r)
        return r, azimuth, polar

    def __init__(self):
        # each quadrant represents the starting radian value
        # running counter clockwise around the circular end
        # of the conicnal search radius
        self.d = self.build_coords(-1, -1, 1)  # DOWN
        self.r = self.build_coords(1, -1, 1)  # RIGHT
        self.u = self.build_coords(1, 1, 1)  # UP
        self.l = self.build_coords(-1, 1, 1)  # LEFT

    def dc(self, c):  # display coords
        return f"radial: {c[0]}, azimuth: {c[1]}, polar: {c[2]}"

    def assign(self, wd: tuple[float, float, float]):
        # wd: Weighted Direction Vector
        # returns the new position
        wdca = self.build_coords(wd[0], wd[1], wd[2])[1]
        if self.d[1] <= wdca < self.r[1]:
            # TODO: Dry this assignment (redundant with init)
            return (-1, -1, 1)
        elif self.r[1] <= wdca < self.u[1]:
            return (1, -1, 1)
        elif self.u[1] <= wdca < self.l[1]:
            return (1, 1, 1)
        elif self.l[1] <= wdca:
            return (-1, 1, 1)
        else:
            raise ValueError("non present radian quadrant")

    def debug_print(self):
        print(f"quadrant d: \n      {self.dc(self.d)}\n")
        print(f"quadrant r: \n      {self.dc(self.r)}\n")
        print(f"quadrant u: \n      {self.dc(self.u)}\n")
        print(f"quadrant l: \n      {self.dc(self.l)}\n")


class Particle:
    def __init__(self, pos: tuple[int, int, int], heading: tuple[float, float, float]):
        self.pos = pos
        self.head = heading
        self.spread = 0.785398  # approx 45 degrees in radians
        self.len = 1
        self.alive = True
        # NOTE: All particles will just have a single pixel grow/sense length

    def search(self, pos: tuple[float, float, float], canvas: np.ndarray) -> float:
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
        ls = self.search(lv, canvas)  # left scalar
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

        qd = Quadrants()
        new_position = qd.assign(wd)

        return new_position, wd

    def draw(self, canvas: np.ndarray):
        # LLM: if block
        if (
            self.pos[0] < 0
            or self.pos[0] >= canvas.shape[0]
            or self.pos[1] < 0
            or self.pos[1] >= canvas.shape[1]
            or self.pos[2] < 0
            or self.pos[2] >= canvas.shape[2]
        ):
            self.alive = False
            return
            # Kills particles at edge of frame
            # HACK: it feels weird that this is in draw.

        draw_val = 1.0
        canvas[int(self.pos[0])][int(self.pos[1])][int(self.pos[2])] = draw_val


sx = 100
sy = 100
sz = 100
fps = 24
rt = 2  # runtime in seconds

decay = 0.99

canvas = np.zeros((sy, sx, sz), dtype=np.float64)  # np uses h*w
particles = []

fcr = 2 * np.pi  # full circle radians

print("random spawning")


def spawn_random():
    for i in range(200):
        particles.append(
            Particle(
                pos=(r.randrange(1, sy), r.randrange(1, sx), r.randrange(1, sz)),
                heading=(r.uniform(0, fcr), r.uniform(0, fcr), r.uniform(0, fcr)),
            )
        )


print("spawned")

spawn_random()
frames = []
# # time steps
for i in range(int(fps * rt)):
    new_particles = []
    for p in particles:
        if p.alive:
            new_pos, new_heading = p.sense_and_rotate(
                canvas
            )  # naming could be better here
            # LLM:
            next_x = p.pos[0] + p.len * new_pos[0]
            next_y = p.pos[1] + p.len * new_pos[1]
            next_z = p.pos[2] + p.len * new_pos[2]

            new_particles.append(
                Particle(pos=(next_x, next_y, next_z), heading=new_heading)
            )
    particles = new_particles
    canvas *= decay
    for p in particles:
        p.draw(canvas)
    frames.append(canvas.copy())
    print(f"frames {i} rendered")

import sys

nv_path = "/Users/joachimpfefferkorn/repos/neurovolume/neurovolume/src"
if nv_path not in sys.path:  # LLM:
    sys.path.insert(0, nv_path)
# VERSION: https://github.com/joachimbbp/neurovolume/tree/59a576bafa2dd9035a08a6c2be40c206c9d53d55
import neurovolume as nv

affine_identity = np.array(
    [  # LLM:
        [1.0, 0.0, 0.0, 0.0],  # x-axis
        [0.0, 1.0, 0.0, 0.0],  # y-axis
        [0.0, 0.0, 1.0, 0.0],  # z-axis
        [0.0, 0.0, 0.0, 1.0],  # homogeneous coord
    ],
    dtype=np.float64,
)

for i, f in enumerate(frames):
    o = f"./output/physarum_nd_{i}.vdb"
    nv.ndarray_to_VDB(f, o, affine_identity)
    print(f"saved {o} to disk")
print("done")
