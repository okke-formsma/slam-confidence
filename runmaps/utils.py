import json, math
from math import sin, cos, atan
import numpy as np

class Container:
    """ Contains the patches data with some slice-and-dice functions.
    """
    def __init__(self, filename):
        self.cache = {}
        with open(filename) as f:
            """ My ugly json printer in VB outputs },\n instead of }] at the end of the file. """
            self.patches = json.loads('[' + f.read()[:-2] + ']')
            for i, p in enumerate(self.patches):
                p['num'] = i

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
            data = [normalize_angle(d) for d in data]

        self.cache[key] = data
        return data

def normalize_angle(angle):
    """ Normalize an angle to be within (-pi, pi]
    """
    angle %= (pi * 2)
    if angle >= pi: angle -= pi * 2
    return angle

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
        return atan(distance((z[0], z[1]), (Z[0], Z[1])) / z[2])

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