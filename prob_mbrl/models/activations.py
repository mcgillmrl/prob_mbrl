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
    def forward(self, x):
        d = int(0.5 * x.shape[-1])
        signs = 2 * torch.arange(2) - 1
        signs = signs.repeat(int(x.shape[-1] / 2 + 1))[:x.shape[-1]]
        x = x * signs
        return torch.cat(
            [x[..., :d].sin(),
             torch.nn.functional.relu(x[..., d:])], -1)
