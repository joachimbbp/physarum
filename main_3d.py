import numpy as np
import math as m
import random as r
import scipy as sp
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
import sys


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


def normalize_vec(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    # LLM: this whole function
    length = np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
    if length == 0:
        return (0, 0, 1)  # default direction
    return (vec[0] / length, vec[1] / length, vec[2] / length)


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


footprint = ndi.generate_binary_structure(3, 1)
scale = 4


class Particle:
    def __init__(
        self, pos: tuple[float, float, float], heading: tuple[float, float, float]
    ):
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

    def rotate_vector(self, vec, axis, angle):
        # LLM: this whole function
        """Rotate a vector around an axis by angle (Rodrigues' rotation)"""
        axis = normalize_vec(axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Rodrigues' rotation formula
        rotated = (
            vec[0] * cos_a
            + (axis[1] * vec[2] - axis[2] * vec[1]) * sin_a
            + axis[0]
            * (axis[0] * vec[0] + axis[1] * vec[1] + axis[2] * vec[2])
            * (1 - cos_a),
            vec[1] * cos_a
            + (axis[2] * vec[0] - axis[0] * vec[2]) * sin_a
            + axis[1]
            * (axis[0] * vec[0] + axis[1] * vec[1] + axis[2] * vec[2])
            * (1 - cos_a),
            vec[2] * cos_a
            + (axis[0] * vec[1] - axis[1] * vec[0]) * sin_a
            + axis[2]
            * (axis[0] * vec[0] + axis[1] * vec[1] + axis[2] * vec[2])
            * (1 - cos_a),
        )
        return normalize_vec(rotated)

    def sense_and_rotate(self, canvas):
        # LLM: Rewrite of this function

        # Create perpendicular vectors for sensing
        # Simple approach: perturb heading in different directions

        # Forward sense position
        fwd_pos = (
            self.pos[0] + self.head[0] * self.len,
            self.pos[1] + self.head[1] * self.len,
            self.pos[2] + self.head[2] * self.len,
        )
        fwd_val = self.search(fwd_pos, canvas)

        # Left sense (rotate heading left around up axis)
        up_axis = (0, 0, 1)
        left_dir = self.rotate_vector(self.head, up_axis, self.spread)
        left_pos = (
            self.pos[0] + left_dir[0] * self.len,
            self.pos[1] + left_dir[1] * self.len,
            self.pos[2] + left_dir[2] * self.len,
        )
        left_val = self.search(left_pos, canvas)

        # Right sense
        right_dir = self.rotate_vector(self.head, up_axis, -self.spread)
        right_pos = (
            self.pos[0] + right_dir[0] * self.len,
            self.pos[1] + right_dir[1] * self.len,
            self.pos[2] + right_dir[2] * self.len,
        )
        right_val = self.search(right_pos, canvas)

        # Calculate weighted direction
        weighted = (
            fwd_val * self.head[0] + left_val * left_dir[0] + right_val * right_dir[0],
            fwd_val * self.head[1] + left_val * left_dir[1] + right_val * right_dir[1],
            fwd_val * self.head[2] + left_val * left_dir[2] + right_val * right_dir[2],
        )

        new_heading = (
            normalize_vec(weighted)
            if (weighted[0] ** 2 + weighted[1] ** 2 + weighted[2] ** 2) > 0
            else self.head
        )

        return new_heading

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
rt = 6  # runtime in seconds

decay = 0.90

num_particles = 200

canvas = np.zeros((sy, sx, sz), dtype=np.float64)  # np uses h*w
particles = []

fcr = 2 * np.pi  # full circle radians

print("random spawning")


def spawn_random():
    # LLM: Rewrite of function
    for i in range(num_particles):
        # Random Cartesian direction
        heading = normalize_vec((r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-1, 1)))
        particles.append(
            Particle(
                pos=(
                    r.randrange(10, sy - 10),
                    r.randrange(10, sx - 10),
                    r.randrange(10, sz - 10),
                ),
                heading=heading,
            )
        )


spawn_random()
print("spawned")
# # time steps
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

nf = fps * rt
for f in range(nf):
    new_particles = []
    print(f"simulating particles on frame {f}/{nf}")
    for p in particles:
        if p.alive:
            new_heading = p.sense_and_rotate(canvas)  # naming could be better here
            # LLM:
            next_x = p.pos[0] + p.len * new_heading[0]
            next_y = p.pos[1] + p.len * new_heading[1]
            next_z = p.pos[2] + p.len * new_heading[2]

            new_particles.append(
                Particle(pos=(next_x, next_y, next_z), heading=new_heading)
            )
            # print(f"particle {i} added")
    particles = new_particles
    canvas *= decay

    print(f"drawing particles for froame {f}/{nf}")
    for p in particles:
        p.draw(canvas)

    # post processing
    c = np.repeat(
        np.repeat(np.repeat(canvas, scale, axis=0), scale, axis=1), scale, axis=2
    )
    c = ndi.grey_dilation(c, size=(8, 8, 8))
    c = gaussian_filter(c, sigma=2)
    # Straight from the docs:
    c = sp.ndimage.grey_erosion(c, footprint=footprint)
    # print(f"particle {i} drawn")

    o = f"./output/physarum_nd_{f}.vdb"
    nv.ndarray_to_VDB(c.copy(), o, affine_identity)
    print(f"saved {o} to disk")
