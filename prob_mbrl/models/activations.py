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
