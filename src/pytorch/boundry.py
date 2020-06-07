import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
import point_cloud_utils as pcu

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='C:/Users/sharon/Documents/Research/data/dataset2019/shapenet/chair/train'
                    , help='data dir [''default: ''../data]')
FLAGS = parser.parse_args()

PARTIAL_PATH = FLAGS.data_path + '/partial/03001627/1a8bbf2994788e2743e99e0cae970928.h5'
COMPLETE_PATH = FLAGS.data_path + '/gt/03001627/1a8bbf2994788e2743e99e0cae970928.h5'


def load_single_file(path):
    fx = h5py.File(path, 'r')
    return np.array(fx['data'])


def plot_pc(pc, pc2=None, show=True, title=""):
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, s=4, cmap='Greens')
    plt.title(title)
    if pc2 is not None:
        v1, v2, v3 = pc2[:, 0], pc2[:, 1], pc2[:, 2]
        ax.scatter3D(v1, v2, v3, s=12, color="r")
    if show:
        plt.show()


def k_nearest_neighbours(data, k=10):
    # calculate square distances between points
    D = distance.squareform(distance.pdist(data))
    # print(np.round(D, 1))
    closest = np.argsort(D, axis=1)
    # For each point, find the k closest points
    k_closest_idx = closest[:, 1:k + 1]
    return data[k_closest_idx]


def calc_centroid(data, k):
    # find k nearest neighbourhood
    return k_nearest_neighbours(data, k).mean()


if __name__ == '__main__':
    points = load_single_file(PARTIAL_PATH)
    complete = load_single_file(COMPLETE_PATH)

    closest_points = k_nearest_neighbours(points, k=10)

    centroids = closest_points.mean(axis=1)
    dist_from_centroid = np.power(points - centroids, 2).sum(axis=1)
    qtl = np.quantile(dist_from_centroid, 0.85)
    idx = np.argwhere(dist_from_centroid >= qtl).flatten()
    outlines = points[idx]
    # print(outlines.shape)
    # plot_pc(points)
    plot_pc(points, outlines)
