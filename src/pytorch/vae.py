import torch
import torch.nn as nn

from src.dataset.shapeDiff import ShapeDiffDataset
from src.pytorch.pointnet import PointNetCls
from src.pytorch.pctools import get_voxel_centers


class Encoder(nn.Module):
    def __init__(self, num_features):
        super(Encoder, self).__init__()
        self.num_features = num_features

        self.cls = PointNetCls(num_features)  # torch.Size([bs, k])

    def forward(self, x):
        """
        :param x: input tensor: torch.Size([bs, 3, num_points])
        :return: probs in torch.Size([bs, num_cubes]),  mu in torch.Size([bs, 3*num_cubes]),
          sigma in torch.Size([bs, 9*num_cubes])
        """

        return torch.sigmoid(self.cls(x))


class VariationalAutoEncoder(nn.Module):

    def __init__(self, n_bins, dev, voxel_sample, threshold):
        """

        :param n_bins:
        :param dev:
        :param voxel_sample:
        :param threshold:
        """
        super(VariationalAutoEncoder, self).__init__()

        self.num_voxels = n_bins ** 3
        self.n_bins = n_bins
        self.threshold = threshold
        self.num_sample_cube = voxel_sample
        self.dev = dev

        self.voxel_centers = get_voxel_centers(self.n_bins).to(dev)
        self.centers = None
        self.voxel_radius = 1 / self.n_bins

        self.encoder = Encoder(num_features=7 * self.num_voxels).float()
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            torch.nn.init.zeros_(m.weight)
            m.bias.data.fill_(0)  # 1 / self.n_bins)

    def _reparameterize(self, centers, sigma):
        """
            This parametrization trick first generates a uniform distribution sample over the unit sphere,
             then shapes the distribution  with the mu and sigma from the encoder.
            This way, we can can calculate the gradient parameterized by this particular random instance.

        :param mu:  Float tensor  in torch.Size([bs, 3*num_cubes])
        :param sigma: Float tensor in torch.Size([bs, 3*num_cubes])
        :return: Float tensor  in torch.Size([bs, num_samples, 3])
        """

        vector_size = (1, self.num_voxels, self.num_sample_cube, 3)

        # sample random standard
        eps = torch.randn(vector_size).to(self.dev)
        eps *= self.voxel_radius * torch.sigmoid(sigma[:, :, None, :]) * 0.1  # .view(-1, 1, 3)
        eps += self.voxel_centers.view(-1, 1, 3) + centers[:, :, None, :]

        return eps

    def forward(self, x, target_gt=None, pred_pc=False):
        """
        :param pred_pc:
        :param target_gt:
        :param x: partial object point cloud
        :return:
        """
        x = x.float()

        s = self.encoder(x)

        probs, mu, sigma = torch.split_with_sizes(s,
                                                  tuple(torch.tensor([1, 3, 3]) * self.num_voxels), axis=1)

        if not pred_pc:
            out = None
        else:
            mu = mu.reshape(mu.shape[0], -1, 3)
            sigma = sigma.reshape(sigma.shape[0], -1, 3)

            # distributing standard normal samples to voxels
            z = self._reparameterize(mu, sigma)  # torch.Size([1, n_bins**3, 20, 3])
            # print(mu)
            if target_gt is None:
                mask = probs[0] > self.threshold  # in shape probs
            else:
                mask = target_gt > 0
            out = z[:, mask]  # torch.Size([high_prob_cubes, 20, 3])
        return out, probs[0]
