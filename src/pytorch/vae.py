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
        self.cls_prob = PointNetCls(k=num_cubes)  # torch.Size([bs, k])
        self.fc_mu = PointNetCls(k=3 * num_cubes)  # torch.Size([bs, 3k])
        self.fc_mat = PointNetCls(k=3 * num_cubes)  # torch.Size([bs, 3k])

    def forward(self, x):
        """
        :param x: input tensor: torch.Size([bs, 3, num_points])
        :return: probs in torch.Size([bs, num_cubes]),  mu in torch.Size([bs, 3*num_cubes]),
          sigma in torch.Size([bs, 9*num_cubes])
        """
        h1 = self.dens(x)

        return F.softmax(self.cls_prob(h1), dim=1), self.fc_mu(h1), self.fc_mat(h1)


# class _Loss_(nn.Module):
#     def __init__(self, coeff):
#         """
#
#         :param coeff:
#         """


class VAELoss(nn.Module):
    def __init__(self, cd_coeff):
        """

        :param coeff: list in length 3
        """
        super(VAELoss, self).__init__()

        self.cd_coeff = cd_coeff

        self.loss = None

    def forward(self, x_diff_pred, x_diff_target):
        """
                gives the batch normalized Variational Error.

        :param x_diff_pred: predicted completion: in shape (bs, num_points (N), 3)
        :param x_diff_target: ground trough completion:  in shape (bs, num_points (M), 3)
        :return: scalar
        """

        # points and points_reconstructed are n_points x 3 matrices
        if x_diff_pred.shape[1] == 0:
            logging.info("Found partial with no positive probability cubes: " + str(x_diff_pred.shape))
            CD = 100
        else:
            CD = chamfer_distance_with_batch(x_diff_pred, x_diff_target, False)

        self.loss = self.cd_coeff * CD


class VariationalAutoEncoder(nn.Module):

    def __init__(self, n_bins, dev, voxel_sample, cf_coeff,
                 threshold, rc_coeff, bce_coeff, regular_method):
        """

        :param n_bins:
        :param dev:
        :param voxel_sample:
        :param cf_coeff:
        :param threshold:
        :param rc_coeff:
        :param bce_coeff:
        :param regular_method:
        """
        super(VariationalAutoEncoder, self).__init__()

        self.num_voxels = n_bins ** 3
        self.n_bins = n_bins
        self.rc_coeff = rc_coeff
        self.bce_coeff = bce_coeff
        self.cd_coeff = cf_coeff
        self.threshold = threshold
        self.num_sample_cube = voxel_sample
        self.dev = dev
        self.regular_method = regular_method

        self.mu = None
        self.sigma = None
        self.probs = None


        self.voxel_centers = get_voxel_centers(self.n_bins).to(dev)
        voxel_radius = 1 / self.n_bins

        self.lower_bound = self.voxel_centers - voxel_radius
        self.upper_bound = self.voxel_centers + voxel_radius

        self.encoder = Encoder(num_cubes=self.num_voxels).float()

        self.fl = FilterLocalization(coeff=self.bce_coeff, threshold=self.threshold)
        self.rc = RegularizedClip(lower=self.lower_bound, upper=self.upper_bound, coeff=self.rc_coeff,
                                  method=self.regular_method)
        self.vloss = VAELoss(cd_coeff=self.cd_coeff)

    def _reparameterize(self, center, scale):
        """
            This reparameterization trick first generates a uniform distribution sample over the unit sphere,
             then shapes the distribution  with the mu and sigma from the encoder.
            This way, we can can calculate the gradient parameterized by this particular random instance.

        :param mu:  Float tensor  in torch.Size([bs, 3*num_cubes])
        :param sigma: Float tensor in torch.Size([bs, 3*num_cubes])
        :return: Float tensor  in torch.Size([bs, num_samples, 3])
        """

        vector_size = (mu.shape[0], self.num_voxels, self.num_sample_cube, 3)

        # sample random standard
        eps = torch.randn(vector_size).to(self.dev)
        eps *= self.sigma.view(scale.shape[0], -1, 1, 3)
        eps += self.mu.view(center.shape[0], -1, 1, 3)

        return eps

    def forward(self, x, x_target, prob_target):
        """

        :param prob_target: frequency in ground trout cubes
        :param x_target: missing regions ground trout point cloud
        :param x: partial object point cloud
        :return:
        """
        x = x.float()

        self.probs, self.mu, self.sigma = self.encoder(x)  # mu, sigma, probs in torch.DoubleTensor

        ##  clipping mu and calculating regulerize loss factor
        self.mu = self.rc(self.mu.view(self.n_bins, self.n_bins, self.n_bins, 3))

        # distributing standard normal samples to voxels
        z = self._reparameterize()  # torch.Size([1, 125, 20, 3])

        # out contains only high probability voxels
        out = self.fl(self.probs, prob_target, z)  # torch.Size([1, 420, 3])

        # chamfer distance between hols point clouds
        self.vloss(out, x_target)

        return out


if __name__ == '__main__':
    bs = 1
    num_points = 250
    resulotion = 5

    train_path = 'C:/Users/sharon/Documents/Research/data/dataset2019/shapenet/train/gt/03001627'

    shapenet = ShapeDiffDataset(train_path, bins=resulotion, dev='cpu')

    train_loader = torch.utils.data.DataLoader(shapenet, 1, shuffle=True)

    x_partial, x_diff, hist = next(iter(train_loader))

    ###########################################
    #
    encoder = Encoder(num_cubes=resulotion ** 3)
    probs, mu, scale = encoder(x_partial.transpose(2, 1))

    print('probs: ', probs.size())  # prob torch.Size([bs, 1000]) view(prob.shape[0], -1, 3)
    print('mu: ', mu.size())  # mu torch.Size([bs, 3000])
    print('scale: ', scale.size())  # scale torch.Size([bs, 9000])
    ###########################################
    #
    # vae = VariationalAutoEncoder(num_cubes=resulotion, dev='cpu')
    #
    # vae_out = vae(x_partial.transpose(2, 1), x_diff, hist.flatten())
    #
    # print("full_out ", vae_out.shape)  # torch.Size([1, num_samples, 3])
    #
    ###########################################

    #### plot centers ####
    # plot_pc([mu_out[0].reshape(-1, 3).detach().numpy()], colors=("black"))
    #
    # z = vae._reparameterize()
    # print("params ", z.shape)  # torch.Size([1, 1000, 100, 3])
