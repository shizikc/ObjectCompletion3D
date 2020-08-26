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


def plot_mesh(ax, x, y, z, ivl, col="red"):
    # mesh cube
    ax.scatter(x, y, z, s=2, color=col)  # left front down corner
    ax.scatter(x, y + ivl, z + ivl, s=2, color=col)  #
    ax.scatter(x, y + ivl, z, s=2, color=col)  #
    ax.scatter(x, y, z + ivl, s=2, color=col)  #
    ax.scatter(x + ivl, y, z + ivl, s=2, color=col)  #
    ax.scatter(x + ivl, y + ivl, z, s=2, color=col)  #
    ax.scatter(x + ivl, y, z, s=2, color=col)  #
    ax.scatter(x + ivl, y + ivl, z + ivl, s=2, color=col)  #


def set_fig(edges):

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    grid_x_ticks_major = edges[0]
    grid_y_ticks_major = edges[1]
    grid_z_ticks_major = edges[2]

    ax.set_xticks(grid_x_ticks_major)
    ax.set_yticks(grid_y_ticks_major)
    ax.set_zticks(grid_z_ticks_major)

    return ax


def plot_pc(pc_lst, colors, show=True, title=""):
    def _plot_layer(pc, col):
        if isinstance(pc, list):
            a, b, c = pc[0], pc[1], pc[2]
        else:
            a, b, c = convert_to_np(pc)
        ax.scatter3D(a, b, c, color=col)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title(title)
    for i, pl in enumerate(pc_lst):
        _plot_layer(pl, colors[i])
    if show:
        plt.show()

def plot_pc_mayavi(pc_lst, colors, show=True, title=""):
    for pc, color in zip(pc_lst, colors):
        mlab.points3d(*pc.T, color=color)
    if show:
        mlab.show()


def vis_prediction(pred_points=None,
                   gt_partial=None,
                   gt_pred=None,
                   voxel_centers=None,
                   voxel_probs=None,
                   # threshold=None,
                   save_to=None):
    if pred_points is not None:
        pred_points = pred_points.reshape(-1, 3)
        mlab.points3d(*pred_points.T, scale_factor=0.02, color=(0.1, 0.9, 0.1))

    if gt_partial is not None:
        gt_partial = gt_partial.reshape(-1, 3)
        mlab.points3d(*gt_partial.T, scale_factor=0.05, color=(1., 0., 0.), opacity=0.7)

    if gt_partial is not None:
        gt_partial = gt_partial.reshape(-1, 3)
        mlab.points3d(*gt_partial.T, scale_factor=0.05, color=(1., 0., 0.), opacity=0.7)

    if voxel_probs is not None:
        assert voxel_centers is not None

        mlab.points3d(*gt_partial.T, scale_factor=0.05, color=(1., 0., 0.), opacity=0.7)
