#!/usr/bin/python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.lib.index_tricks import s_
from utils import Container, distance, Angle, plot_displacement_map

"""
Plots
location error
rotation error
covariance trace
map
"""

def main(filename = '2012-05-31 IRO2012-Pre2/patches.json'):
    patches = Container(filename)
    plot_error(patches)
    #plt.tight_layout()
    plt.show()


def plot_error(patches):
    diffs = [distance((p['groundtruth']['x'], p['groundtruth']['y']), (p['slam']['x'], p['slam']['y'])) 
             for p in patches]

    diff_yaw = [abs(Angle.diff(p['groundtruth']['yaw'], p['slam']['yaw'])) for p in patches]

    trace = [cov[0]+cov[4]+cov[8] for cov in patches['covariance']]
    num = [p['num'] for p in patches]

    fig = plt.figure()
    gs = gridspec.GridSpec(5,1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax4 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[3:5, 0])

    ax1.plot(num, diffs)
    ax1.set_ylabel('location error')

    ax4.plot(num, diff_yaw)
    ax4.set_ylabel('rotation error')

    ax2.semilogy(num, trace)
    ax2.set_ylabel('covariance trace')
    ax2.set_xlabel('patch #')

    plot_displacement_map(patches, ax3)
    #gs.tight_layout(fig)


if __name__ == '__main__':
    main()