import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Module

from src.pytorch.pointnet import PointNetfeat, PointNetDenseCls, PointNetCls
import torch.nn.functional as F


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        # Submodules
        self.conv_0 = torch.nn.Conv1d(size_in, size_h, 1)
        self.conv_1 = torch.nn.Conv1d(size_h, size_out, 1)

        # Initialization
        torch.nn.init.zeros_(self.conv_1.weight)

    def forward(self, x):
        """

        :param x: torch.Size([batch_size, size_in, num_points])
        :return: x + dx : torch.Size([batch_size, size_in, num_points])
        """
        dx = self.conv_0(F.relu(x))
        dx = self.conv_1(F.relu(dx))

        return x + dx


class ResNetBlocks(nn.Module):
    """
    PointNet Encoder

    """

    def __init__(self):
        super(ResNetBlocks, self).__init__()

        self.feat = PointNetfeat(global_feat=False)

        self.conv1 = torch.nn.Conv1d(1088, 1024, 1)

        self.block0 = ResnetBlockFC(1024)
        self.block1 = ResnetBlockFC(1024)
        self.block2 = ResnetBlockFC(1024)
        self.block3 = ResnetBlockFC(1024)
        self.block4 = ResnetBlockFC(1024)

        self.conv2 = torch.nn.Conv1d(1024, 512, 1)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        """
        :param x: in torch.Size([batch_size, num_points, 3])
        :return: pred in torch.Size([batch_size, 1024])
        """
        x = x.transpose(2, 1)
        x, _, _ = self.feat(x)  # torch.Size([batch_size, 1088, num_points])
        x = F.relu(self.bn1(self.conv1(x)))  # torch.Size([batch_size, 1024, num_points])
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)  # torch.Size([batch_size, 1024, num_points])
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x


class HDM(Module):
    def __init__(self, out_dim):
        super(HDM, self).__init__()
        self.cls = PointNetCls(k=out_dim)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.cls(x)[0]  # torch.Size([bs, 8000])
        x = torch.sigmoid(x)
        return x




if __name__ == '__main__':
    bs = 10
    n_points = 1740
    sim_data = Variable(torch.rand(bs, n_points, 3))
    print(sim_data.shape)

    cls = HDM(20 ** 3)
    out = cls(sim_data)
    print('class', out.size())  # class torch.Size([10, 20**3])
    print('class', out[0].sum())  # class torch.Size([10, 20**3])

