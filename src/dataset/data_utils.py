import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import mayavi.mlab as mlab
import numpy as np


def plot_hist3d(hist, color=(1.0, 1.0, 1.0), lower_lim=-1, upper_lim=1, lst=None):
    if lst is None:
        lst = [np.linspace(lower_lim, upper_lim, 10) for i in range(3)]
    x, y, z = np.meshgrid(lst[0], lst[1], lst[2])
    mlab.points3d(x, y, z, hist, color=color, mode='cube')


def convert_to_np(pc):
    if isinstance(pc, list):
        return pc[0], pc[1], pc[2]
    return pc[:, 0], pc[:, 1], pc[:, 2]


def plot_pc(pc, vec=None, show=True, title=""):
    a, b, c = convert_to_np(pc)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(a, b, c, s=4, color='black')
    plt.title(title)
    if vec is not None:
        v1, v2, v3 = convert_to_np(vec)
        ax.scatter3D(v1, v2, v3, s=4, color="r")
    if show:
        plt.show()

