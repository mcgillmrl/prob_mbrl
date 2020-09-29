import torch


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class Exp(torch.nn.Module):
    def forward(self, x):
        return (-0.5 * x**2).exp()


class Sin(torch.nn.Module):
    def forward(self, x):
        return x.sin()


class SinLU(torch.nn.Module):
    def __init__(self):
        super(SinLU, self).__init__()
        self.thr = torch.nn.Threshold(0, 0)

    def forward(self, x):
        return self.thr(x) - self.thr(-x).sin()


class hhSinLU(torch.nn.Module):
    def __init__(self):
        super(hhSinLU, self).__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        signs = 2 * torch.arange(2) - 1
        signs = signs.repeat(int(x.shape[-1] / 2 + 1))[:x.shape[-1]]
        x = x * signs
        x1, x2 = x.chunk(2, -1)
        x = torch.cat([x1.sin(), self.relu(x2)], -1)
        return x
