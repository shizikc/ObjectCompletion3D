import logging

import torch
import torch.nn as nn

import torch.nn.functional as F

from src.chamfer_distance.chamfer_distance import chamfer_distance_with_batch
from src.dataset.shapeDiff import ShapeDiffDataset
from src.pytorch.region_select import FilterLocalization
from src.pytorch.pointnet import PointNetDenseCls, PointNetCls
from src.pytorch.range_bounds import RegularizedClip
from src.pytorch.pctools import get_voxel_centers


class Encoder(nn.Module):
    def __init__(self, num_cubes):
        super(Encoder, self).__init__()
        self.num_cubes = num_cubes

        # input: torch.Size([bs, 3, num_points])
        self.dens = PointNetDenseCls(k=3)  # torch.Size([bs, 3, num_points])

        # input: torch.Size([bs, 3, num_points])
        self.cls = PointNetCls(k=7*num_cubes)  # torch.Size([bs, k])

    def forward(self, x):
        """
        :param x: input tensor: torch.Size([bs, 3, num_points])
        :return: probs in torch.Size([bs, num_cubes]),  mu in torch.Size([bs, 3*num_cubes]),
          sigma in torch.Size([bs, 9*num_cubes])
        """
        h1 = self.dens(x)

        return F.sigmoid(self.cls(h1))  # , self.fc_mu(h1), F.logsigmoid(self.fc_mat(h1))


class VariationalAutoEncoder(nn.Module):

    def __init__(self, n_bins, dev, voxel_sample, threshold, regular_method):
        """

        :param n_bins:
        :param dev:
        :param voxel_sample:
        :param threshold:
        :param regular_method:
        """
        super(VariationalAutoEncoder, self).__init__()

        self.num_voxels = n_bins ** 3
        self.n_bins = n_bins
        self.threshold = threshold
        self.num_sample_cube = voxel_sample
        self.dev = dev
        self.regular_method = regular_method

        self.voxel_centers = get_voxel_centers(self.n_bins).to(dev)
        self.voxel_radius = 1 / self.n_bins

        self.encoder = Encoder(num_cubes=self.num_voxels).float()
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            torch.nn.init.zeros_(m.weight)
            m.bias.data.fill_(0)  # 1 / self.n_bins)

    def _reparameterize(self):
        """
            This reparameterization trick first generates a uniform distribution sample over the unit sphere,
             then shapes the distribution  with the mu and sigma from the encoder.
            This way, we can can calculate the gradient parameterized by this particular random instance.

        :param mu:  Float tensor  in torch.Size([bs, 3*num_cubes])
        :param sigma: Float tensor in torch.Size([bs, 3*num_cubes])
        :return: Float tensor  in torch.Size([bs, num_samples, 3])
        """

        vector_size = (1, self.num_voxels, self.num_sample_cube, 3)

        # sample random standard
        eps = torch.randn(vector_size).to(self.dev)
        eps *= self.voxel_radius  # .view(-1, 1, 3)
        eps += self.voxel_centers.view(-1, 1, 3)

        return eps

    def forward(self, x):
        """
        :param x: partial object point cloud
        :return:
        """
        x = x.float()

        s = self.encoder(x)  # mu, sigma, probs in torch.DoubleTensor

        probs, mu, sigma = torch.split_with_sizes(s, tuple(torch.tensor([1, 3, 3]) * self.num_voxels), axis=1)

        mu = mu.reshape(mu.shape[0], -1, 3)
        sigma = sigma.reshape(sigma.shape[0], -1, 3)

        # distributing standard normal samples to voxels
        z = self._reparameterize()  # torch.Size([1, n_bins**3, 20, 3])
        print(z.shape)
        mask = probs > self.threshold  # in shape probs
        out = z[:, mask]  # torch.Size([high_prob_cubes, 20, 3])
        print(out.shape)
        out = out.view(out.shape[0], -1, 3)
        return out, probs


if __name__ == '__main__':
    bs = 1
    num_points = 250
    resulotion = 5

    train_path = 'C:/Users/sharon/Documents/Research/data/dataset2019/shapenet/train/gt/03001627'

    shapenet = ShapeDiffDataset(train_path, bins=resulotion, dev='cpu')

    train_loader = torch.utils.data.DataLoader(shapenet, 1, shuffle=True)

    x_partial, x_diff, hist = next(iter(train_loader))

    ###########################################

    encoder = Encoder(num_cubes=resulotion ** 3)
    probs = encoder(x_partial.transpose(2, 1))

    print('probs: ', probs.size())  # prob torch.Size([bs, 1000]) view(prob.shape[0], -1, 3)

    ###########################################

    vae = VariationalAutoEncoder(n_bins=5, dev="cpu", voxel_sample=20, threshold=0.01, regular_method="abs")

    vae_out, probs_out = vae(x_partial.transpose(2, 1))

    ###########################################

    #### plot centers ####
    # plot_pc([mu_out[0].reshape(-1, 3).detach().numpy()], colors=("black"))
    #
    # z = vae._reparameterize()
    # print("params ", z.shape)  # torch.Size([1, 1000, 100, 3])
