import numpy as np


# SOURCE: https://en.wikipedia.org/wiki/Spherical_coordinate_system?utm_source=chatgpt.com
# spherical coordiante systems specifies a given point in 3D space by using:
# - a radial distance (connects the coordinate to the origin)
# - the polar angle (between radial dist and a given polar axis)
# - the azimuth angle (angle of rotation of the radial line around the polar axis), (theta in wolfram)


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
        self.d = self.build_coords(-1, -1, 1)  # DOWN
        self.r = self.build_coords(-1, 1, 1)  # RIGHT
        self.u = self.build_coords(1, 1, 1)  # UP
        self.l = self.build_coords(1, -1, 1)  # LEFT

    def debug_print(self):
        print(f"quadrant d: {self.d} of type {type(self.d)}")
        print(f"quadrant r: {self.r} of type {type(self.r)}")
        print(f"quadrant u: {self.u} of type {type(self.u)}")
        print(f"quadrant l: {self.l} of type {type(self.l)}")


q = Quadrants()
q.debug_print()
