#!~/anaconda3/envs/teb/bin/python
import os
import sys
sys.path.append(os.path.abspath('.'))

import numpy as np
from scipy import interpolate

def _grid_reduction(map2D: np.ndarray, xscale=1., yscale=1.):
    assert xscale <= 1.
    assert yscale <= 1.
    shape = map2D.shape
    new_x = int(shape[0]*xscale)
    new_y = int(shape[1]*yscale)
    new_map = np.zeros((new_x, new_y), dtype=np.float32)
    for i in range(new_x):
        for j in range(new_y):
            x1 = int(i/xscale)
            x2 = int((i+1)/xscale)
            y1 = int(j/yscale)
            y2 = int((j+1)/yscale)
            block = map2D[x1:x2,y1:y2]
            new_map[i][j] = np.mean(block)
    return new_map

def _grid_magnification(map2D: np.ndarray, xscale=1., yscale=1.):
    assert xscale >= 1.
    assert yscale >= 1.
    shape = map2D.shape
    new_x = int(shape[0]*xscale)
    new_y = int(shape[1]*yscale)
    new_map = np.zeros((new_x, new_y), dtype=np.float32)
    x_grid = range(shape[0])
    y_grid = range(shape[1])
    f = interpolate.interp2d(x_grid, y_grid, map2D.T, kind='linear')
    for i in range(new_x):
        for j in range(new_y):
            x = i/xscale
            y = j/yscale
            new_map[i][j] = f(x, y)
    return new_map

def resize_grid_2D(map2D: np.ndarray, new_shape):
    shape = map2D.shape
    assert len(new_shape) == 2
    assert len(shape) == 2
    xscale = new_shape[0] / shape[0]
    yscale = new_shape[1] / shape[1]

    if xscale >= 1.:
        new_map = _grid_magnification(map2D, xscale=xscale)
    else:
        new_map = _grid_reduction(map2D, xscale=xscale)
    
    if yscale >= 1.:
        new_map = _grid_magnification(new_map, yscale=yscale)
    else:
        new_map = _grid_reduction(new_map, yscale=yscale)
    
    return new_map

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/maps/example1.png'
    img_array = mpimg.imread(path)
    grey_map = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
    new_shape = (500, 505)
    new_img = resize_grid_2D(grey_map, new_shape)
    plt.figure()
    plt.imshow(new_img)
    plt.show()
    