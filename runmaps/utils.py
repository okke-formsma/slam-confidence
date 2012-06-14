import json
from math import pi, sqrt

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
        """
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
        self.cache[key] = data
        return data


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


def distance((x1, y1), (x2, y2)):
    """ Returns euclidian disntace between self and other """
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
