import argparse
import os
import random
from enum import Flag
from pathlib import Path

import h5py
import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset

from src.dataset.data_utils import plot_pc, plot_hist3d

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/data', help='data dir [''default: ''../data]')

FLAGS = parser.parse_args()

GT_PATH = 'C:/Users/sharon/Documents/Research/data/dataset2019/shapenet/chair/val/gt/03001627/'
# PATH = FLAGS.data_path
LOWER_LIM = -1
UPPER_LIM = 1


def load_single_file(path, data_name="data"):
    fx = h5py.File(path, 'r')
    return np.array(fx[data_name])


## UTILITY FUNCTIONS TO CREATE AND SAVE DATASETS#
def data_writer(dirname, n_files=-1):
    """
    create a training data set by intersecting the complete object with a hyperplane,
    such that the partial object obtained is 70%-85% of the complete.
    3 folders are written: partial, diff and hist_label using path.

    :param dirname:
    :param folder_replace: list : which folder in path to replace, default ot gt. the replacement will be
    "partial", "diff", "hist_labels"
    :param n_files:
    :param path: complete data path folder
    :return:
    """

    Path(dirname.replace('gt', 'partial_sub_group')).mkdir(parents=True, exist_ok=True)
    Path(dirname.replace('gt', 'diff')).mkdir(parents=True, exist_ok=True)
    Path(dirname.replace('gt', 'hist_labels')).mkdir(parents=True, exist_ok=True)
    # read the complete full path to a list
    list_files = os.listdir(dirname)
    if n_files < 0:
        n_files = len(list_files)

    for i in range(n_files):
        fn = os.path.join(dirname, list_files[i])
        # read complete
        x_complete = load_single_file(fn)
        # create partial
        x_partial = create_partial_from_complete(x_complete)
        # save partial
        with h5py.File(fn.replace('gt', 'partial_sub_group'), 'w') as hf:
            hf.create_dataset("data", data=x_partial)
        # create diff
        x_diff = create_diff_point_cloud(x_complete, x_partial)
        # save diff
        with h5py.File(fn.replace('gt', 'diff'), 'w') as hf:
            hf.create_dataset("data", data=x_diff)
        # create labels
        H, edges = create_hist_labels(x_diff)
        # save labels
        with h5py.File(fn.replace('gt', 'hist_labels'), 'w') as hf:
            hf.create_dataset("edges", data=edges)
            hf.create_dataset("hist", data=H)


def create_hist_labels(diff_set):
    """

    :param diff_set:
    :return:
    H: ndarray The multidimensional histogram of sample x. See normed and weights for the different possible semantics.
    edges: list A list of D arrays describing the bin edges for each dimension.
    """
    r = (LOWER_LIM, UPPER_LIM)
    return np.histogramdd(diff_set, bins=10, range=(r, r, r), density=True)


def create_diff_point_cloud(pc1, pc2):
    """
    extracts all points in pc1 but not in pc2
    :param pc1: numpy of shape(num_points, 3) - complete object
    :param pc2: numpy of shape(num_points, 3) - partial object
    :return:
    """
    _, indices = np.unique(np.concatenate([pc2, pc1]), return_index=True, axis=0)
    indices = indices[indices >= pc2.shape[0]] - pc2.shape[0]
    return pc1[indices]


def create_partial_from_complete(complete):
    """
    create a partial object with 70%-85% unique points.
    The returned object is then added with randomly duplicated points to contain exactly 1740 points
    :param complete:
    :return:
    """
    rng = np.random.default_rng()
    s = 0
    mn, mx = complete.min(), complete.max()
    # 70% (int(0.7 * 2048)=1433) < |partial object| < 85% (int(0.85 * 2048)=1740)
    while s < 1433 or s > 1740:
        a, b, c = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)
        cond = a * complete[:, 0] + b * complete[:, 1] + c * complete[:, 2] > random.uniform(mn, mx)
        s = sum(cond)

    x_partial_tmp = complete[cond]
    idx = rng.choice(s, 1740 - s, replace=False)
    return np.concatenate((x_partial_tmp, x_partial_tmp[idx, :]))


class ShapeDiffDataset(Dataset):
    """shapenet partial dataset"""

    def __init__(self, partial_dir, replace_dir='partial_sub_group',
                 transform=None, val=True):
        """

        :param partial_dir: string : Directory with all the partial shapes. should contain replce_dir
        :param transform:
        :param replace_dir: directory name stating the partial data set, replace with diff/gt/hist_labels accordingly
        """
        self.replace_dir = replace_dir
        self.partial_path = partial_dir  # input path
        if val:
            self.partial_path = partial_dir.replace('train', 'val')
        self.fn_list = os.listdir(self.partial_path)

        self.transform = transform

    def __len__(self):
        return len(self.fn_list)

    def __getitem__(self, idx):
        in_path = os.path.join(self.partial_path, self.fn_list[idx])
        label_path = in_path.replace(self.replace_dir, 'hist_labels')

        x = load_single_file(in_path)
        hist = load_single_file(label_path, "hist")

        return x, hist.flatten()


if __name__ == '__main__':
    pass
    # shapenetDataset = ShapeDiffDataset(PATH)
    # x_partial, hist  = shapenetDataset[1]
    # plot_pc(x_partial)
    # plot_hist3d(hist)
    # data_writer(GT_PATH, n_files=20)

#     x_complete = load_single_file(COMPLETE_PATH)
#     x_partial = load_single_file(COMPLETE_PATH.replace('gt', 'partial_sub_group'))
#     x_diff = load_single_file(COMPLETE_PATH.replace('gt', 'diff'))
#
#     # x_partial = create_partial_from_complete(x_complete)
#     # x_diff = set_diff_point_cloud(x_complete, x_partial)
#     # H, edges = set_hist_labels(x_diff)
#     # # plot_pc(x_partial, x_diff)

#     x, y, z = convert_to_np(x_diff)
#     x1, y1, z1 = convert_to_np(x_partial)
#     #
#     mlab.points3d(x, y, z, color=(0.2, 0.4, 0.5), scale_factor=.02)
#     mlab.points3d(x1, y1, z1, color=(0.9, 0.7, 0.2), scale_factor=.02)
#     mlab.show()
