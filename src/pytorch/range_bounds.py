import torch
from torch.nn import Module
import torch.nn.functional as F


class RegularizedClip(Module):
    def __init__(self, lower, upper, coeff, method="abs"):
        super(RegularizedClip, self).__init__()
        assert method in ["abs", "square"]
        self.method = method
        self.coeff = coeff
        self.lower = lower
        self.upper = upper
        self.loss = None

    def forward(self, x):
        c = torch.max(x, self.lower.view(x.shape))
        c = torch.min(c, self.upper.view(x.shape)).view(1, -1)

        r = self.upper - self.lower

        mean = (self.upper + self.lower) * 0.5
        y = (x - mean) / r * 2.  # normalize - mean 0

        y = F.relu(torch.abs(y) - r)  # y punishes outside of [-1, 1]

        if self.method == "square":
            y = torch.pow(y, 2.)
        self.loss = y.sum() * self.coeff

        return c


if __name__ == "__main__":
    e0 = torch.arange(-1, 1, 1)
    e1 = e0 + 1

    xv0, yv0, zv0 = torch.meshgrid(e0, e0, e0)  # each is (20,20,20)
    lower_bound = torch.stack((xv0, yv0, zv0), dim=3).float()

    xv1, yv1, zv1 = torch.meshgrid(e1, e1, e1)  # each is (20,20,20)
    upper_bound = torch.stack((xv1, yv1, zv1), dim=3).float()

    x = torch.randn(upper_bound.shape)

    rc = RegularizedClip(coeff=1., lower=lower_bound, upper=upper_bound)
    c = rc(x)

    # d = torch.nn.Conv1d(10, 10, 1)
    # d.weight.data.zero_()
    # d.bias.data[...] = torch.linspace(0, 10, 10)
    #
    # y = d(x)
    # z = rc(y)
    # o = z.sum()
    # reg = rc.loss
    # o += reg
    #
    # d.zero_grad()
    # o.backward()
    #
    # # print(d.bias.grad.reshape(-1))
    # print(reg)
