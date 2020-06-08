import torch
from torch.nn import Module


class HDM(Module):
    def __init__(self, out_dim):
        super(HDM, self).__init__()
        self.out_dim = out_dim
        self.init = torch.zeros((self.out_dim,))
        self.lin1 = torch.nn.Linear(self.out_dim, self.out_dim)

    def forward(self, input):
        print("init : " + str(self.init.dtype))
        print("input : " + str(input.dtype))
        x = self.lin1(self.init)
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    hdm_model = HDM(1000)
    for i in range(20):
        hdm_model(torch.DoubleTensor([1.2,2.2,.2]))
