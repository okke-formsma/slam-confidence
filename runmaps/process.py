#!/usr/bin/python
from utils import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

maps = {
    'quad': '2012-05-31 IRO2012-Pre2/patches_quadwsm.json',
    'wsm':  '2012-05-31 IRO2012-Pre2/patches.json',
}

def main(filename = None):
    if filename is None:
        filename = maps['wsm']
    patches = Container(filename)
    #plot_pitch_roll(patches)
    #plot_confidence(patches)
    #plot_slam_covariance(patches)
    #plot_paths(patches)
    plot_error_metrics(patches)
    plt.show()


def plot_slam_covariance(patches):
    """ Plots the slam path and associated covariance ellipsis.
    """
    fig = plt.figure(figsize=(6, 12))
    gs = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_path(patches, source='slam', ax=ax1, plot_numbers=True)
    plot_error_ellipsis(patches, ax1)
    ax1.set_ylim(ax1.get_ylim()[::-1]) #inverse y axis
    plt.tight_layout()
    plt.savefig('slam_covariance.pdf')

def plot_error_metrics(patches):
    """ Plots the three error metrics
    """
    fig = plt.figure(figsize=(8, 3))
    gs = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])

    det = np.abs(np.array([np.linalg.det(cov) for cov in patches['cov']]))
    trace = np.abs(np.array([np.trace(cov) for cov in patches['cov']]))
    missing = -np.isfinite(det) * 1
    #print missing, np.sum(missing)
    ax1.semilogy(det, label="Determinant")
    ax1.semilogy(trace, 'g', label="Trace")
    ax1.semilogy(missing, 'r.', label="NaN")

    ax1.set_ylim(bottom=1, top=10**9.9)
    ax1.set_ylabel('Absolute value')
    ax1.set_xlabel('patch #')

    ax1.legend(loc=2)
    
    plt.tight_layout()
    plt.savefig('error_metrics.pdf')

def plot_paths(patches):
    """ Plots the groundtruth, ins and slam paths in an image and saves it
    """
    fig = plt.figure(figsize=(3,5))
    gs = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0,0])
    plot_path(patches, source='groundtruth', ax=ax1, plot_numbers=True)
    plot_path(patches, source='ins', ax=ax1)
    plot_path(patches, source='slam', ax=ax1)
    ax1.set_ylim(ax1.get_ylim()[::-1]) #inverse y axis
    ax1.legend(loc=2)
    plt.tight_layout()
    plt.savefig('paths.pdf')

def plot_confidence(patches):
    diffs = []
    for x1, y1, x2, y2 in zip(patches['groundtruth.x'], patches['groundtruth.y'], patches['ins.x'], patches['ins.y']):
        diffs.append(distance((x1, y1), (x2, y2)))

    diff_yaw = [Angle.diff(yaw1, yaw2)
                for yaw1, yaw2 in zip(patches['ins.yaw'], patches['groundtruth.yaw'])]

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

    trace = [cov.trace() for cov in patches['cov']]

    fig = plt.figure()
    gs = gridspec.GridSpec(5,2)
    loc_ax = fig.add_subplot(gs[0, 0])
    yaw_ax = fig.add_subplot(gs[1, 0])
    ins_ax = fig.add_subplot(gs[2, 0])
    log_conf_ax = fig.add_subplot(gs[3, 0])
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

    conf_ax.plot(num, trace)
    log_conf_ax.semilogy(num, trace)
    log_conf_ax.set_ylabel('trace log')
    conf_ax.set_ylabel('trace')
    conf_ax.set_xlabel('patch #')

    plot_displacement_map(patches, map_ax)


if __name__ == '__main__':
    main()