import argparse
import os
import random
from pathlib import Path
import logging
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset.data_utils import plot_pc


logging.getLogger().setLevel(logging.INFO)


# PATH = FLAGS.data_path
LOWER_LIM = -1
UPPER_LIM = 1


def load_single_file(path, data_name="data"):
    fx = h5py.File(path, 'r')
    return np.array(fx[data_name])


# UTILITY FUNCTIONS TO CREATE AND SAVE DATASETS#
def data_writer(dirname, n_files=-1):
    """
    create a training data set by intersecting the complete object with a hyperplane,
    such that the partial object obtained is 70%-85% of the complete.
    3 folders are written: partial, diff and hist_label using path.

    :param dirname: gt folder
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
        # print("x_complete: ", x_complete.shape)
        # create partial
        x_partial = create_partial_from_complete(x_complete)
        # print("x_partial: ", x_partial.shape)
        # save partial
        with h5py.File(fn.replace('gt', 'partial_sub_group'), 'w') as hf:
            hf.create_dataset("data", data=x_partial)
        # create diff
        x_diff = create_diff_point_cloud(x_complete, x_partial)
        # print("x_diff: ", x_diff.shape)
        # save diff
        with h5py.File(fn.replace('gt', 'diff'), 'w') as hf:
            hf.create_dataset("data", data=x_diff)
        # create labels
        H, edges = create_hist_labels(x_diff, args.bins)
        # print("edges: ", edges.shape)
        # save labels
        with h5py.File(fn.replace('gt', 'hist_labels'), 'w') as hf:
            hf.create_dataset("edges", data=edges)
            hf.create_dataset("hist", data=H)

    logging.info("Done writing files to : % s", dirname)


def create_hist_labels(diff_set, bins):
    """

    :param diff_set: tuple containing (X, Y, Z)
    :return:
    H: ndarray The multidimensional histogram of sample x. See normed and weights for the different possible semantics.
    edges: list A list of D arrays describing the bin edges for each dimension.
    """
    r = (LOWER_LIM, UPPER_LIM)
    H = np.histogramdd((diff_set[:, 0], diff_set[:, 1], diff_set[:, 2]), bins=bins, range=(r, r, r))
    return H[0] / diff_set.shape[0], H[1]


def create_diff_point_cloud(pc1, pc2):
    """
    extracts all points in pc1 but not in pc2
    :param pc1: numpy of shape(num_points, 3) - complete object
    :param pc2: numpy of shape(num_points, 3) - partial object
    :return: numpy of shape(diff num_points, 3)
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
    return torch.tensor(np.concatenate((x_partial_tmp, x_partial_tmp[idx, :])))


class ShapeDiffDataset(Dataset):
    """shapenet partial dataset"""

    def __init__(self, path, bins, dev):
        """

        :param root_path: string : Root directory of structure: root
                                                                    diff
                                                                    gt
                                                                    hist_labels
                                                                    partial_sub_group
                                                                            object_id
                                                                                files.h5
        :param object_id:
        """
        self.path = path
        self.bins = bins
        self.dev = dev
        self.fn_list = os.listdir(self.path)

    def __len__(self):
        return len(self.fn_list)

    def __getitem__(self, idx):
        in_path = os.path.join(self.path, self.fn_list[idx])

        x_complete = load_single_file(in_path)
        x_partial = create_partial_from_complete(x_complete)
        x_diff = create_diff_point_cloud(x_complete, x_partial)
        H, edges = create_hist_labels(x_diff, self.bins)

        return x_partial.to(self.dev), torch.tensor(x_diff).to(self.dev), torch.tensor(H).to(self.dev)


if __name__ == '__main__':
    train_path = 'C:/Users/sharon/Documents/Research/data/dataset2019/shapenet/train/gt/03001627'

    shapenet = ShapeDiffDataset(train_path, 20)
    x_partial, x_diff, hist = shapenet[0]
    plot_pc([x_partial, x_diff], colors=("black", "red"))
