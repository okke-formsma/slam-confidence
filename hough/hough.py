from math import hypot, pi
import numpy as np
from scipy.misc import imread, imsave, imresize
from subprocess import call
import time

def hough(original, theta_resolution=360, rho_resolution=360):
    """ Calculate Hough transform. """
    original_x, original_y = im.shape
    hough_data = np.zeros((theta_resolution, rho_resolution), dtype=int)

    rho_delta = hypot(original_x, original_y) / (rho_resolution / 2)
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


im = imread('lines.png', flatten=True)
im = imresize(im, (im.shape[0], im.shape[1]))
t = time.time()
im = -hough(im).transpose()
print time.time() - t
imsave('hough.png', im)
call(["open", "hough.png"])