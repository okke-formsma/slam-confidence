from scipy.misc import imread, imsave, imresize
import numpy as np
import glob

def main():
    file_filter = r"./2012-05-31 IRO2012-Pre2/maps/*jury*.png"
    for path in glob.glob(file_filter):
        im = imread(path)
        #Only keep blue channel - red and green are 0 in the 'unknown' parts.
        im = im[:, :, 2]
        im = crop(im)
        imsave(path[:-11] + '.png', im)

def crop(im):
    """  crop image to minimum bounding boxes.
    """
    x_usage = np.where(np.sum(255 - im, 1) > 0)
    print x_usage
    x_min, x_max = np.min(x_usage), np.max(x_usage) + 1
    y_usage = np.where(np.sum(255 - im, 0) > 0)
    y_min, y_max = np.min(y_usage), np.max(y_usage) + 1
    return im[x_min:x_max, y_min:y_max]

if __name__ == '__main__':
    main()