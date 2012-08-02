from math import hypot, pi
from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave, imresize
from subprocess import call
import time

def hough_transform(original, theta_resolution=180, rho_resolution=250):
    """ Calculate Hough transform. """
    original_x, original_y = original.shape
    hough_data = np.zeros((theta_resolution, rho_resolution), dtype=int)

    rho_delta = 2 * hypot(original_x, original_y) / rho_resolution
    theta_delta = pi / theta_resolution

    theta_index = np.arange(theta_resolution, dtype=int)
    theta_cos = np.cos(theta_index * theta_delta)
    theta_sin = np.sin(theta_index * theta_delta)

    for x in xrange(original_x):
        xcos = x * theta_cos
        for y in xrange(original_y):
            if original[x, y] == 255: continue #background is white, all else is a line.
            rho = xcos + y * theta_sin
            rho_index = np.array(np.floor(rho / rho_delta + 0.5), dtype=int) + rho_resolution / 2
            hough_data[theta_index, rho_index] += 1

    return hough_data

def hough_spectrum(hough_data):
    """ Calculates the normalized hough spectrum for hough data.
    """
    spectrum = np.sum(hough_data ** 2, 1, dtype=float)
    return spectrum / np.max(spectrum)

def plot_hough_spectrum(im):
    rho_resolution=250
    theta_resolution=180
    ht = hough_transform(im, theta_resolution, rho_resolution)
    hs = hough_spectrum(ht)

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1)

    ax1 = fig.add_subplot(gs[0, 0])
    imgplt = ax1.imshow(ht.transpose(), origin='lower', cmap='hot', aspect='auto')
    ax1.set_xlabel('theta / pi')
    ax1.set_ylabel('rho (normalized)')
    #set proper theta labels
    ax1.set_xticks(np.linspace(0, theta_resolution, 4, endpoint=False))
    ax1.set_xticklabels(['%s pi' % n for n in np.linspace(0, 1, 4, endpoint=False)])
    #set proper rho labels
    ax1.set_yticks(np.linspace(0, rho_resolution, 5))
    ax1.set_yticklabels(np.linspace(-1, 1, 5))

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(hs)


    plt.show()

def plot_hough_rotate(im, im2):
    ht = hough_transform(im)
    ht2 = hough_transform(im2)
    hs = hough_spectrum(ht)
    hs2 = hough_spectrum(ht2)
    corr = np.correlate(np.tile(hs, 2), hs2)

    fig = plt.figure()
    gs = gridspec.GridSpec(3, 1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(hs)
    ax1.set_ylabel('hough spectrum 1')
    ax1.set_xlabel(r'theta')

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(hs2)
    ax2.set_ylabel('hough spectrum 2')
    ax2.set_xlabel(r'theta')

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(corr)
    ax3.set_ylabel('cross-correlation')
    ax3.set_xlabel(r'theta')

    plt.show()

#plot_hough_rotate(imread('lines.png', flatten=True), imread('lines.png', flatten=True).rot90())
plot_hough_spectrum(imread('lines.png', flatten=True))
#imsave('hough.png', -ht.rot90())
#call(["open", "hough.png"])