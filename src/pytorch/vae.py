import torch
from torch.autograd import Variable
import torch.nn as nn

import torch.nn.functional as F

# from src.chamfer_distance import ChamferDistance
from src.pytorch.pointnet import PointNetDenseCls, PointNetCls


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 9, 1)
        self.conv2 = torch.nn.Conv1d(9, 3, 1)
        self.bn1 = nn.BatchNorm1d(9)

    def forward(self, z):
        """

        :param z: torch.Size([bs, 3, num_points])
        :return: torch.Size([bs, 3, num_points])
        """
        z = F.relu(self.bn1(self.conv1(z)))  # : torch.Size([bs, 9, num_points])
        x = self.conv2(z)  # : torch.Size([bs, 3, num_points])
        return z.transpose(2, 1).contiguous()  # torch.Size([bs, num_points, 3])


class Encoder(nn.Module):
    def __init__(self, num_cubes):
        super(Encoder, self).__init__()
        self.num_cubes = num_cubes
        # input: torch.Size([bs, 3, num_points])
        self.dens = PointNetDenseCls(k=3)  # torch.Size([bs, 3, num_points])

        # input: torch.Size([bs, 3, num_points])
        self.cls_prob = PointNetCls(k=num_cubes)  # torch.Size([bs, k])
        self.fc_mu = PointNetCls(k=3 * num_cubes)  # torch.Size([bs, 3k])
        self.fc_mat = PointNetCls(k=9 * num_cubes)  # torch.Size([bs, 9k])

    def forward(self, x):
        """

        :param x: input tensor: torch.Size([bs, 3, num_points])
        :return: probs in torch.Size([bs, num_cubes]),  mu in torch.Size([bs, 3*num_cubes]),
          sigma in torch.Size([bs, 9*num_cubes])
        """
        bs = x.shape[0]
        h1 = self.dens(x)
        return F.softmax(self.cls_prob(h1)), self.fc_mu(h1), self.fc_mat(h1)


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='mean')
        # self.chamfer_dist = ChamferDistance()

    def forward(self, prob_pred, prob_target, x_diff_pred, x_diff_target):
        """
                gives the batch normalized Variational Error.

        :param prob_pred: in shape (bs, K) where K- number of cubes
        :param prob_target: in shape (bs, K) where K- number of cubes
        :param x_diff_pred: predicted completion: in shape (bs, num_points (N), 3)
        :param x_diff_target: ground trouth completion:  in shape (bs, num_points (M), 3)
        :return: scalar
        """

        BCE = self.bce_loss(prob_pred, prob_target)

        # points and points_reconstructed are n_points x 3 matrices
        # dist1, dist2 = self.chamfer_dist(x_complete, points_reconstructed)
        # CD = (torch.mean(dist1)) + (torch.mean(dist2))

        return BCE #+ CD


class VariationalAutoEncoder(nn.Module):

    def __init__(self, samples_cube=100):
        super(VariationalAutoEncoder, self).__init__()
        self.samples_cube = samples_cube
        self.encoder = Encoder(num_cubes=10 ** 3)
        self.decoder = Decoder()

    def forward(self, x):
        probs, mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        return self.decoder(z), mu, sigma

    def reparameterize(mu, sigma):
        """
            This reparameterization trick first generates a uniform distribution sample over the unit sphere,
             then shapes the distribution  with the mu and sigma from the encoder.
            This way, we can can calculate the gradient parameterized by this particular random instance.

        :param mu:  Float tensor  in torch.Size([bs, 3*num_cubes])
        :param sigma: Float tensor in torch.Size([bs, 9*num_cubes])
        :return: Float tensor in torch.Size([bs, samples_cube * num_samples])
        """

        vector_size = sigma.size()

        # sample random uniformly
        eps = Variable(torch.FloatTensor(vector_size).normal_())
        return eps.mul(sigma).add_(mu)


if __name__ == '__main__':

    in_data = Variable(torch.rand(32, 3, 250))
    encoder = Encoder(num_cubes=10 ** 3)
    prob, mu, scale = encoder(in_data)

    print('prob', prob.size()) #prob torch.Size([32, 1000])
    print('mu', mu.size()) #mu torch.Size([32, 3000])
    print('scale', scale.size()) #scale torch.Size([32, 9000])

    out_data = Variable(torch.rand(32, 3, 500))

    decoder = Decoder()
    out = decoder(out_data)
    print("out", out.shape) # torch.rand(32, 3, 500)

