#!/usr/bin/python
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from random import random

def main(filename = '2012-05-31 IRO2012-Pre2/patches.json'):
    patches = load_patches(filename)
    #plot_displacement_map(patches)
    plot_error(patches)
    plt.show()

def load_patches(filename):
    with open(filename) as f:
        """ My ugly json printer in VB outputs },\n instead of }] at the end of the file. """
        patches = json.loads('[' + f.read()[:-2] + ']')
    return patches

def plot_error(patches):
    diffs = []
    for p in patches:
        diffs.append(((p['groundtruth']['x'] - p['slam']['x']) ** 2 +
                      (p['groundtruth']['x'] - p['slam']['x']) ** 2) ** 0.5)

    diff_rot = [abs(p['groundtruth']['rot'] - p['slam']['rot']) for p in patches]

    covar = [float(p['avgcovariancedeterminant']) for p in patches]

    fig = plt.figure()
    gs = gridspec.GridSpec(3,2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax4 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[:, 1])

    ax1.plot(diffs)
    ax1.set_ylabel('location error')

    ax4.plot(diff_rot)
    ax4.set_ylabel('rotation error')

    ax2.plot(covar)
    ax2.set_ylabel('covariance determinant \nof correspondence matrix')
    ax2.set_xlabel('patch #')

    plot_displacement_map(patches, ax3)
    gs.tight_layout(fig)


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
    displacement_segments_x = []
    displacement_segments_y = []
    angles_g = []
    angles_s = []
    for gx_, gy_, _gw, sx_, sy_, _sw in zip(gx, gy, gw, sx, sy, sw):
        displacement_segments_x.extend([gx_ + random(), sx_ + random(), None])
        displacement_segments_y.extend([gy_ + random(), sy_ + random(), None])

    ax.plot(displacement_segments_x, displacement_segments_y, '-', color=(0.1, 0.9, 0.1, 0.3))
    ax.plot(gx, gy, 'b+-')
    ax.plot(sx, sy, 'r+-')

    for num in range(0, len(sx), 10):
        ax.annotate(str(num), xy=(sx[num], sy[num]), xytext=(sx[num], sy[num]))

    ax.set_ylim(ax.get_ylim()[::-1]) #inverse y axis

#plt.savefig('displacement_map.png')
    #plt.savefig('displacement_map.pdf')
    #plt.show()
    
if __name__ == '__main__':
    main()