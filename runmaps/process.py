#!/usr/bin/python
from utils import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats

maps = {
    'iro': '2012-05-31 IRO2012-Pre2/patches.json',
    'ned': '2012-08-07 nl pre1/patches.json',
    'smoke': '2012-08-09 IranOpen2012-SemiFinal-withSmoke/patches.json'
}

def main(filename = None):
    if filename is None:
        filename = maps['smoke']
    patches = Container(filename)

    #plot_pitch_roll(patches)
    #plot_confidence(patches)
    plot_slam_covariance(patches)
    plot_paths(patches)
    plot_error_metrics(patches)
    get_piece_ranges(patches)

    #plot_error_metrics_scatter(patches)
    plt.tight_layout()

    plt.show()

def get_piece_ranges(patches):
    matches = np.array(patches['matches'])

    last_break = 0
    print np.where(matches == 0)[0]
    for b in np.where(matches == 0)[0]:
        if last_break != b-1: #no consecutive runs
            print last_break+1, b-1
        last_break = b
    print last_break+1, len(matches)


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
    """ Plots the three error metrics on a time axis
    """
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    det = np.abs(np.array([np.linalg.det(cov) for cov in patches['cov']]))
    trace = np.abs(np.array([np.trace(cov) for cov in patches['cov']]))
    missing = -np.isfinite(det) * 1
    #print missing, np.sum(missing)
    ax1.semilogy(det, label="Determinant")
    ax1.semilogy(trace, 'g', label="Trace")
    ax1.semilogy(missing, 'r.', label="NaN")

    ax1.set_ylim(bottom=1, top=10**9.9)
    ax1.set_xlim(right=len(det))
    #ax1.set_ylabel('')
    ax1.legend(loc=2)

    ax2.plot(patches['matches'], label='matches')
    ax2.legend(loc=2)
    ax2.set_xlabel('t')
    ax2.set_xlim(right=len(det))

    #plt.tight_layout()
    plt.savefig('error-metrics.pdf')


def plot_error_metrics_scatter(patches):
    """ Plots the three error metrics in relation to each other
    """
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0, 2])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 0])

    det = np.abs(np.array([np.trace(cov) for cov in patches['cov']]))
    trace = np.abs(np.array([np.linalg.det(cov) for cov in patches['cov']]))
    matches = np.array(patches['matches'])
    missing = -np.isfinite(det) * 1

    ax1.semilogy(matches, trace, 'bo')
    ax1.semilogy(matches, missing, 'ro')
    ax1.set_xlabel('matches')
    ax1.set_ylabel('trace')
    ax1.set_ylim(bottom=1, top=10 ** 8)

    ax2.semilogy(matches, det, 'bo')
    ax2.semilogy(matches, missing, 'ro')
    ax2.set_xlabel('matches')
    ax2.set_ylabel('determinant')
    ax2.set_ylim(bottom=1)

    ax3.loglog(trace, det, 'bo')
    ax3.loglog(missing, missing, 'ro')
    ax3.set_xlabel('trace')
    ax3.set_ylabel('determinant')
    ax3.set_ylim(bottom=1)

    detn, tracen, matchesn = np.nan_to_num(det), np.nan_to_num(trace), np.nan_to_num(matches)

    print 'trace det', scipy.stats.spearmanr(tracen, detn)
    print 'matches det', scipy.stats.spearmanr(matchesn, detn)
    print 'matches trace', scipy.stats.spearmanr(matchesn, tracen)

    mask = np.where(matches != 0)
    print matches[mask]
    num = 10
    trace_max = set(np.argsort(trace[mask])[:num])
    det_max = set(np.argsort(det[mask])[:num])
    matches_min = set(np.argsort(matches[mask])[-num:])

    print "trace", trace_max
    print "det", det_max
    print "matches", matches_min
    print 'trace det', len(trace_max & det_max)
    print 'matches det', len(matches_min & det_max)
    print 'matches trace', len(matches_min & trace_max)


    plt.tight_layout()
    plt.savefig('error-measures-scatter.pdf')
    plt.show()

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