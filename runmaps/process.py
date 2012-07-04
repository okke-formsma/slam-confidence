#!/usr/bin/python
from utils import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

maps = {
    'quad': '2012-05-31 IRO2012-Pre2/patches_quadwsm.json',
    'wsm': '2012-05-31 IRO2012-Pre2/patches_wsm.json',
}

def main(filename = maps['wsm']):
    patches = Container(filename)
    plot_pitch_roll(patches)
    #plot_confidence(patches)
    plt.show()


def plot_confidence(patches):
    diffs = []
    for x1, y1, x2, y2 in zip(patches['groundtruth.x'], patches['groundtruth.y'], patches['slam.x'], patches['slam.y']):
        diffs.append(distance((x1, y1), (x2, y2)))

    diff_yaw = [Angle.diff(yaw1, yaw2)
                for yaw1, yaw2 in zip(patches['slam.yaw'], patches['groundtruth.yaw'])]

    fig = plt.figure()
    gs = gridspec.GridSpec(4, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax4 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[3:5, 0])

    num = patches['num']
    ax1.plot(num, diffs)
    ax1.set_ylabel('location error')

    ax4.plot(num, diff_yaw)
    ax4.set_ylabel('rotation error')

    ax2.plot(num, patches['avgcovariancedeterminant'], color=(1, 0, 0, 0.8))
    ax2.set_ylabel('Confidence')
    ax2.set_xlabel('patch #')

    plot_displacement_map(patches, ax3)

def plot_pitch_roll(patches):
    diffs = []
    for x1, y1, x2, y2 in zip(patches['groundtruth.x'], patches['groundtruth.y'], patches['slam.x'], patches['slam.y']):
        diffs.append(distance((x1, y1), (x2, y2)))

    diff_yaw = [abs(Angle.diff(yaw1, yaw2))
                for yaw1, yaw2 in zip(patches['slam.yaw'], patches['groundtruth.yaw'])]

    attack_ins = [Orientation(**ins).attack() for ins in patches['ins']]
    attack_gt = [Orientation(**gt).attack() for gt in patches['groundtruth']]

    trace = [cov[0,0] + cov[1,1] for cov in patches['cov']]

    fig = plt.figure()
    gs = gridspec.GridSpec(5,2)
    loc_ax = fig.add_subplot(gs[0, 0])
    yaw_ax = fig.add_subplot(gs[1, 0])
    ins_ax = fig.add_subplot(gs[2, 0])
    conf_ax = fig.add_subplot(gs[4, 0])
    map_ax = fig.add_subplot(gs[:, 1])

    num = patches['num']
    loc_ax.plot(num, diffs)
    loc_ax.set_ylabel('location error')

    yaw_ax.plot(num, diff_yaw)
    yaw_ax.set_ylabel('rotation error')

    ins_ax.plot(num, attack_ins, 'r-')
    ins_ax.plot(num, attack_gt, 'b-')
    ins_ax.set_ylabel('attack')

    conf_ax.semilogy(num, trace)
    conf_ax.set_ylabel('trace')
    conf_ax.set_xlabel('patch #')

    plot_displacement_map(patches, map_ax)


if __name__ == '__main__':
    main()