import numpy as np


"""
conversions between different coordinate systems

"""


def cart2cylinder(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    z = z
    return r, theta, z


def cylinder2cart(r, theta, z):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = z
    return x, y, z
