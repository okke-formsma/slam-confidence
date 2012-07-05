import json, math
from math import sin, cos, atan, pi
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import linalg
from numpy.linalg.linalg import LinAlgError

class Container:
    """ Contains the patches data with some slice-and-dice functions.
    """
    def __init__(self, filename):
        self.cache = {}
        with open(filename) as f:
            """ My ugly json printer in VB outputs },\n instead of }] at the end of the file. """
            self.patches = json.loads('[' + f.read()[:-3] + ']')
            for i, p in enumerate(self.patches):
                p['num'] = i
                p['cov'] = np.reshape(p['covariance'], (3,3))[:2,:2] #from 3D to 2D

    def __iter__(self):
        """ Returns an iterator over all patches
        """
        return self.patches.__iter__()

    def __getitem__(self, key):
        """ Makes dictionary-like access of all members possible. For example: obj['num'], or obj['slam.x']
        Also allows direct patch access through integer-access: obj[41]
        """
        if isinstance(key, (int, long)):
            return self.patches[key]

        if key in self.cache:
            return self.cache[key]

        keys = key.split('.')
        if len(keys) == 1:
            #base case: no dots.
            base = self.patches
        else:
            #recurse with subkey
            base = self['.'.join(keys[:-1])]

        data = [b[keys[-1]] for b in base]

        if keys[-1] in ('roll', 'yaw', 'pitch'):
            data = [Angle.normalize(d) for d in data]

        self.cache[key] = data
        return data

def distance((x1, y1), (x2, y2)):
    """ Returns euclidian disntace between self and other """
    return math.hypot(x1 - x2, y1 - y2)

class Orientation(object):
    def __init__(self, x=0, y=0, z=0, yaw=0, pitch=0, roll=0):
        self.pose = np.array([x, y, z], np.float)
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

    def attack(self):
        """ The attack is the angle between (global-coordinates) Z-axis and the (local) z-axis.
        """
        Z = np.array([0,0,1], np.float)
        full_rotation = self.roll_matrix().dot(self.pitch_matrix())
        z = full_rotation.dot(Z)
        return abs(atan(distance((z[0], z[1]), (Z[0], Z[1])) / z[2])) #can be negative?

    def yaw_matrix(self):
        y = self.yaw
        return np.array(
            [
                [cos(y), -sin(y), 0],
                [sin(y), cos(y), 0],
                [0, 0, 1],
            ], np.float)

    def pitch_matrix(self):
        p = self.pitch
        return np.array(
            [
                [cos(p), 0, sin(p)],
                [0, 1, 0],
                [-sin(p), 0, cos(p)],
            ], np.float)

    def roll_matrix(self):
        r = self.roll
        return np.array(
            [
                [1, 0, 0],
                [0, cos(r), -sin(r)],
                [0, sin(r),  cos(r)],
            ], np.float)

class Angle:
    @classmethod
    def normalize(cls, angle):
        return angle % (pi * 2)

    @classmethod
    def diff(cls, first, second):
        """ Returns normalized difference between self and other in (-pi, pi]"""
        diff = first - second
        while diff > pi:   diff -= 2 * pi
        while diff <= -pi: diff += 2 * pi
        return diff

def error_ellipse(patch):
    """ Calculates width, height and angle (radians) for error ellipse.
    """
    cov = linalg.inv(patch['cov']) #meters to millimeters
    try:
        lengths = sorted(np.linalg.eigvals(cov), reverse=True)
    except LinAlgError:
        return Ellipse(xy=(patch['slam']['x'], patch['slam']['y']),
                       width=500, height=500, angle=0,
                       color=(1,1,0))
    scalefactor = 2.4477 #95% interval
    dxy = cov[0, 1]
    dx = cov[0, 0]
    dy = cov[1, 1]
    rotation = 0.5 * (math.atan((2 * dxy) / ((dx ** 2) - (dy ** 2))))
    rotation += patch['slam']['yaw']

    # sometimes we get negative eigenvalues (which shouldn't happen) so we abs them.
    width = math.sqrt(abs(lengths[0])) * scalefactor
    height = math.sqrt(abs(lengths[1])) * scalefactor

    if dy > dx:
        width, height = height, width

    print width, height, rotation
    return Ellipse(xy=(patch['slam']['x'], patch['slam']['y']),
                width=width * 1000, height=height * 1000, angle=math.degrees(rotation),
                color=(1,0,1), alpha=0.3)


def plot_error_ellipsis(patches, ax=None):
    """ Plots error ellipses from patches on axis ax. """
    if ax is None:
        ax = plt.figure()

    for p in patches:
        ax.add_artist(error_ellipse(p))

def plot_displacement_map(patches, ax=None):
    """ Plots a map showing the groundtruth (blue) and slam (red) trails of the robots,
    with the same patches connected by a green line. """

    if ax is None:
        ax = plt.figure()

    gx = patches['groundtruth.x']
    gy = patches['groundtruth.y']
    gw = patches['groundtruth.yaw']
    sx = patches['slam.x']
    sy = patches['slam.y']
    sw = patches['slam.yaw']

    ax.axis('equal')
    displacement_segments = dict(x=[], y=[]) # x = list of x1, x2, None, ... of list segments that go from x1 to x2
    angles_g = dict(x=[], y=[])
    angles_s = dict(x=[], y=[])

    for gx_, gy_, gw_, sx_, sy_, sw_ in zip(gx, gy, gw, sx, sy, sw):
        displacement_segments['x'].extend([gx_, sx_, None])
        displacement_segments['y'].extend([gy_, sy_, None])
        angles_g['x'].extend([gx_, gx_ + 200 * cos(gw_), None])
        angles_g['y'].extend([gy_, gy_ + 200 * sin(gw_), None])
        angles_s['x'].extend([sx_, sx_ + 200 * cos(sw_), None])
        angles_s['y'].extend([sy_, sy_ + 200 * sin(sw_), None])

    # all plotting with x- and y-values swapped
    ax.plot(displacement_segments['x'], displacement_segments['y'], '-', color=(0.1, 0.9, 0.1, 0.3))
    ax.plot(angles_g['x'], angles_g['y'], '-', color=(0, 0, 0, 0.8))
    ax.plot(angles_s['x'], angles_s['y'], '-', color=(0, 0, 0, 0.8))
    ax.plot(gx, gy, '+-', color=(0, 0, 1, 0.5))
    ax.plot(sx, sy, '+-', color=(1, 0, 0, 0.8), linewidth=2)

    for p in patches:
        if p['num'] % 10 == 0:
            ax.annotate(str(p['num']), xy=(p['slam']['x'], p['slam']['y']), xytext=(p['slam']['x'], p['slam']['y']))

    ax.set_ylim(ax.get_ylim()[::-1]) #inverse y axis

    plot_error_ellipsis(patches, ax)
