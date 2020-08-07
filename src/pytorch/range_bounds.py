import torch
from torch.nn import Module
import torch.nn.functional as F


class RegularizedClip(Module):
    def __init__(self, coeff=0.01, method="abs"):
        super(RegularizedClip, self).__init__()
        assert method in ["abs", "square"]
        self.method = method
        self.coeff = coeff
        self.loss = None

    def forward(self, x, lower, upper):
        c = torch.clamp(x, lower, upper)
        r = upper - lower
        mean = (upper + lower) * 0.5
        y = (x - mean) / r * 2.
        y = F.relu(torch.abs(y) - 1.)
        if self.method == "square":
            y = torch.square(y)
        self.loss = y.sum() * self.coeff
        return c


if __name__ == "__main__":
    import numpy as np

    d = torch.nn.Conv1d(10, 10, 1, )
    d.weight.data.zero_()
    d.bias.data[...] = torch.linspace(0, 10, 10)
    rc = RegularizedClip(coeff=5., method="square")
    x = torch.randn([1,10,1])
    y = d(x)
    z = rc(y, 3., 7.)
    o = z.sum()
    reg = rc.loss
    o += reg
    d.zero_grad()
    o.backward()

    print(d.bias.grad.reshape(-1))

