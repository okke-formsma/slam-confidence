import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from random import random

def main():
    with open('patches.json') as f:
        """ My ugly json printer in VB outputs },\n instead of }] at the end of the file. """
        patches = json.loads('[' + f.read()[:-2] + ']')
        
    plot_displacement_map(patches)
    
    diffs = []
    for p in patches:
        diffs.append(  #euclidian distance
            (
                (p['groundtruth']['x'] - p['slam']['x']) ** 2 +
                (p['groundtruth']['x'] - p['slam']['x']) ** 2 
            ) ** 0.5
        )

    covar = [float(p['avgcovariancedeterminant']) for p in patches]
    
    gs = gridspec.GridSpec(2,2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[:, 1])
    
    ax1.plot(diffs)
    ax1.set_ylabel('location error')
    
    ax2.plot(covar)
    ax2.set_ylabel('covariance determinant \nof correspondence matrix')
    ax2.set_xlabel('patch #')
    
    ax3.scatter(diffs, covar)
    ax3.set_xlabel('location error')
    ax3.set_ylabel('covariance')
    
    plt.tight_layout()
    plt.show()

def plot_displacement_map(patches):
    """ Plots a map showing the groundtruth (blue) and slam (red) trails of the robots, 
    with the same patches connected by a green line. """
    gx = [p['groundtruth']['x'] for p in patches]
    gy = [-p['groundtruth']['y'] for p in patches]
    sx = [p['slam']['x'] for p in patches]
    sy = [-p['slam']['y'] for p in patches]

    plt.axis('equal')
    displacement_segments_x = []
    displacement_segments_y = []
    for gx_, gy_, sx_, sy_ in zip(gx, gy, sx, sy):
        displacement_segments_x.extend([gx_ + random(), sx_ + random(), None])
        displacement_segments_y.extend([gy_ + random(), sy_ + random(), None])
        plt.plot(displacement_segments_x, displacement_segments_y, '-', color=(0.1, 0.9, 0.1)) 
        
    plt.plot(gx, gy, 'b+-')
    plt.plot(sx, sy, 'r+-')
    plt.show()
    
if __name__ == '__main__':
    main()