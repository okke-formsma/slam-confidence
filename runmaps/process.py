#!/usr/bin/python
import json
from math import pi, sqrt, sin, cos
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

maps = [
    '2012-05-31 IRO2012-Pre2/patches.json',
    '2012-06-14 IRO2012-Pre2/patches.json',
    ]

def main(filename = maps[1]):
    patches = load_patches(filename)
    plot_error(patches)
    #plt.tight_layout()
    plt.show()

def load_patches(filename):
    with open(filename) as f:
        """ My ugly json printer in VB outputs },\n instead of }] at the end of the file. """
        patches = json.loads('[' + f.read()[:-2] + ']')

    i = 0
    for p in patches:
        p['num'] = i
        i += 1

    return patches

def plot_error(patches):
    diffs = []
    for p in patches:
        diffs.append(distance((p['groundtruth']['x'], p['groundtruth']['y']), (p['slam']['x'], p['slam']['y'])))

    diff_yaw = [abs(Angle.diff(p['groundtruth']['yaw'], p['slam']['yaw'])) for p in patches]

    roll_ins = [float(p['ins']['roll']) for p in patches]
    pitch_ins = [float(p['ins']['pitch']) for p in patches]
    roll_g = [float(p['groundtruth']['roll']) for p in patches]
    pitch_g = [float(p['groundtruth']['pitch']) for p in patches]
    num = [p['num'] for p in patches]

    fig = plt.figure()
    gs = gridspec.GridSpec(6,1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax4 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[3, 0])
    ax3 = fig.add_subplot(gs[4:7, 0])

    ax1.plot(num, diffs)
    ax1.set_ylabel('location error')

    ax4.plot(num, diff_yaw)
    ax4.set_ylabel('rotation error')

    ax2.plot(num, pitch_ins, color=(1, 0, 0, 0.8))
    ax2.plot(num, pitch_g, color=(0, 0, 1, 0.8))
    ax2.set_ylabel('pitch')

    ax5.plot(num, roll_ins, color=(1, 0, 0, 0.8))
    ax5.plot(num, roll_g, color=(0, 0, 1, 0.8))
    ax5.set_ylabel('roll')
    ax5.set_xlabel('patch #')

    plot_displacement_map(patches, ax3)
    #gs.tight_layout(fig)


def plot_displacement_map(patches, ax=None):
    """ Plots a map showing the groundtruth (blue) and slam (red) trails of the robots, 
    with the same patches connected by a green line. """

    if ax is None:
        ax = plt.figure()

    gx = [p['groundtruth']['x'] for p in patches]
    gy = [p['groundtruth']['y'] for p in patches]
    gw = [p['groundtruth']['yaw'] for p in patches]
    sx = [p['slam']['x'] for p in patches]
    sy = [p['slam']['y'] for p in patches]
    sw = [p['slam']['yaw'] for p in patches]

    ax.axis('equal')
    displacement_segments = dict(x=[], y=[]) # x = list of x1, x2, None, ... of list segments that go from x1 to x2
    angles_g = dict(x=[], y=[])
    angles_s = dict(x=[], y=[])
    for gx_, gy_, gw_, sx_, sy_, sw_ in zip(gx, gy, gw, sx, sy, sw):
        displacement_segments['x'].extend([gx_, sx_, None])
        displacement_segments['y'].extend([gy_, sy_, None])
        angles_g['x'].extend([gx_, gx_ + 200*cos(gw_), None])
        angles_g['y'].extend([gy_, gy_ + 200*sin(gw_), None])
        angles_s['x'].extend([sx_, sx_ + 200*cos(sw_), None])
        angles_s['y'].extend([sy_, sy_ + 200*sin(sw_), None])

    # all plotting with x- and y-values swapped
    ax.plot(displacement_segments['y'], displacement_segments['x'], '-', color=(0.1, 0.9, 0.1, 0.3))
    ax.plot(angles_g['y'], angles_g['x'], '-', color=(0, 0, 0, 0.8))
    ax.plot(angles_s['y'], angles_s['x'], '-', color=(0, 0, 0, 0.8))
    ax.plot(gy, gx, '+-', color=(0, 0, 1, 0.5))
    ax.plot(sy, sx, '+-', color=(1, 0, 0, 0.8), linewidth=2)

    for p in patches:
        if p['num'] % 10 == 0:
            ax.annotate(str(p['num']), xy=(p['slam']['y'], p['slam']['x']), xytext=(p['slam']['y'], p['slam']['x']))

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

if __name__ == '__main__':
    main()