import torch
from torch.nn import Module
import torch.nn.functional as F


class RegularizedClip(Module):
    def __init__(self, lower, upper, coeff=1., method="abs"):
        super(RegularizedClip, self).__init__()
        assert method in ["abs", "square"]
        self.method = method
        self.coeff = coeff
        self.lower = lower
        self.upper = upper
        self.loss = None

    def forward(self, x):

        c = torch.min(x, self.lower)
        c = torch.max(c, self.upper).view(1, -1)

        r = self.upper - self.lower
        mean = (self.upper + self.lower) * 0.5
        y = (x - mean) / r * 2. # normelize - mean 0
        y = F.relu(torch.abs(y) - 1.) # y in [-1,1]
        if self.method == "square":
            y = torch.pow(y, 2.)
        self.loss = y.sum() * self.coeff

        return c





if __name__ == "__main__":
    import numpy as np
    x = torch.randn([1, 10, 1])
    rc = RegularizedClip(coeff=5., method="square")

    d = torch.nn.Conv1d(10, 10, 1, )
    d.weight.data.zero_()
    d.bias.data[...] = torch.linspace(0, 10, 10)
    y = d(x)

    z = rc(y, 3., 7.)
    o = z.sum()
    reg = rc.loss
    o += reg

    # d.zero_grad()
    # o.backward()

    print(d.bias.grad.reshape(-1))