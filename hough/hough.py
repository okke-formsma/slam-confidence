from math import hypot, pi, radians
from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave, imresize
from scipy.ndimage.interpolation import rotate
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

def x_y_spectrum(im):
    return np.sum(im != 255, 0), np.sum(im != 255, 1)

def plot_hough(im):
    rho_resolution = 250
    theta_resolution = 180
    ht = hough_transform(im, theta_resolution, rho_resolution)

    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1)

    ax1 = fig.add_subplot(gs[0, 0])
    imgplt = ax1.imshow(ht.transpose(), origin='lower', cmap='hot', aspect='auto')
    ax1.set_ylabel('rho (normalized)')
    #set proper rho labels
    ax1.set_yticks(np.linspace(0, rho_resolution, 5))
    ax1.set_yticklabels(np.linspace(-1, 1, 5))
    ax1.set_xticks(np.linspace(0, theta_resolution, 4, endpoint=False))
    ax1.set_xticklabels(['%s pi' % n for n in np.linspace(0, 1, 4, endpoint=False)])
    ax1.set_xlabel('theta / pi')

    plt.show()

def plot_hough_spectrum(im):
    rho_resolution=250
    theta_resolution=180
    ht = hough_transform(im, theta_resolution, rho_resolution)
    hs = hough_spectrum(ht)

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1)

    ax1 = fig.add_subplot(gs[0, 0])
    imgplt = ax1.imshow(ht.transpose(), origin='lower', cmap='hot', aspect='auto')
    ax1.set_ylabel('rho (normalized)')
    #no theta labels
    ax1.set_xticks([])
    #set proper rho labels
    ax1.set_yticks(np.linspace(0, rho_resolution, 5))
    ax1.set_yticklabels(np.linspace(-1, 1, 5))

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(hs)
    #set proper theta labels
    ax2.set_xticks(np.linspace(0, theta_resolution, 4, endpoint=False))
    ax2.set_xticklabels(['%s pi' % n for n in np.linspace(0, 1, 4, endpoint=False)])
    ax2.set_xlim(0, len(hs)-1)
    ax2.set_xlabel('theta / pi')
    ax2.set_ylabel('hough spectrum')

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


def plot_after_rotation(im, rotation, im2, rotation2):
    im = rotate(im, angle=rotation, mode='nearest')
    im2 = rotate(im2, angle=rotation2, mode='nearest')
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(im, cmap='gray')
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(im2, cmap='gray')

def plot_x_y_correlation(im, im2, phi, theta):
    im = rotate(im, angle=phi, mode='nearest')
    im2 = rotate(im2, angle=phi+theta, mode='nearest')
    x, y = x_y_spectrum(im)
    x2, y2 = x_y_spectrum(im2)
    corr_x = np.correlate(x, x2)

    fig = plt.figure()
    gs = gridspec.GridSpec(3, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x2)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot()

def plot_x_y_spectrum(im):
    x, y = x_y_spectrum(im)
    fig = plt.figure()
    gs = gridspec.GridSpec(2,2)
    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(y, range(len(y)))
    ax1.set_ylim(0, len(y))
    ax1.invert_yaxis()
    ax1.set_title('Y-spectrum')
    ax1.set_ylabel('y')
    ax1.set_xlabel('occupied cells')
    ax2 = fig.add_subplot(gs[1,1])
    ax2.plot(x)
    ax2.set_xlim(0, len(x))
    ax2.set_title('X-spectrum')
    ax2.set_ylabel('occupied cells')
    ax2.set_xlabel('x')
    ax3 = fig.add_subplot(gs[0,1])
    ax3.imshow(im, cmap="gray", aspect="auto")
    ax3.set_title('Map')

im1 = imread('rooms.png', flatten=True)
im2 = imread('rooms-partial.png', flatten=True)

#plot_after_rotation(imread('rooms.png', flatten=True), 90, imread('rooms-partial.png', flatten=True), 169+90)
plot_x_y_correlation(im1, im2, 90, 169)
#plot_hough_rotate(imread('rooms.png', flatten=True), imread('rooms-partial.png', flatten=True))
#plot_hough_spectrum(imread('rooms.png', flatten=True))
#plot_x_y_spectrum(imread('rooms-rotated-full.png', flatten=True))
#imsave('hough.png', -ht.rot90())
#call(["open", "hough.png"])

print 'done'
plt.show()