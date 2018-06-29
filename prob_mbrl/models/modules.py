import torch

from torch import nn
from torch.nn import Parameter


class BDropout(nn.Dropout):
    """
        Extends the base Dropout layer by adding a regularizer as derived by
        Gal and Ghahrahmani "Dropout as a Bayesian Approximation" (2015)

    """

    def __init__(self, rate=0.5,
                 name=None,
                 regularizer_scale=1.0,
                 **kwargs):
        super(BDropout, self).__init__(**kwargs)
        self.name = name
        self.regularizer_scale = regularizer_scale
        self.rate = Parameter(torch.tensor(rate), requires_grad=False)
        self.p = 1 - self.rate
        self.noise = Parameter(
            torch.bernoulli(1.0 - self.rate), requires_grad=False)

    def weights_regularizer(self, weights):
        self.p = 1 - self.rate
        return 0.5*(self.regularizer_scale**2)*(self.p*weights**2).sum()

    def biases_regularizer(self, biases):
        return 0.5*(self.regularizer_scale**2)*(biases**2).sum()

    def resample(self):
        self.update_noise(self.noise)

    def update_noise(self, x):
        self.p = 1 - self.rate
        self.noise.data = torch.bernoulli(self.p.expand(x.shape))

    def forward(self, x, resample=True):
        if x.shape != self.noise.shape or resample:
            self.update_noise(x)
        return x*self.noise


class CDropout(BDropout):
    def __init__(self, rate=0.5,
                 name=None,
                 regularizer_scale=1.0,
                 temperature=0.1,
                 **kwargs):
        super(CDropout, self).__init__(
            rate, name, regularizer_scale, **kwargs)
        self.temp = Parameter(torch.tensor(temperature), requires_grad=False)
        self.logit_p = Parameter(
            -torch.log(1.0/torch.tensor(1 - self.rate) - 1.0))

    def weights_regularizer(self, weights):
        p = self.logit_p.sigmoid()
        reg = 0.5*(self.regularizer_scale**2)*(p*weights**2).sum()
        reg -= -(1-p)*(1-p).log() - p*p.log()
        return reg

    def update_noise(self, x):
        self.noise.data = torch.rand_like(x)

    def forward(self, x, resample=True):
        if x.shape != self.noise.shape or resample:
            self.update_noise(x)

        p = self.logit_p.sigmoid()
        concrete_p = p.log() - (1-p).log()\
            + self.noise.log() - (1 - self.noise).log()
        concrete_noise = (concrete_p/self.temp).sigmoid()
        return x*concrete_noise


class BSequential(nn.modules.Sequential):
    " An extension to sequential that allows for controlling resampling"

    def __init__(self, *args):
        super(BSequential, self).__init__(*args)

    def resample(self):
        for module in self._modules.values():
            if isinstance(module, BDropout):
                module.resample()

    def forward(self, input, resample=True, **kwargs):
        for module in self._modules.values():
            if isinstance(module, BDropout):
                input = module(input, resample=resample)
            else:
                input = module(input)
        return input

    def regularization_loss(self):
        modules = self._modules.values()
        reg_loss = 0
        for i, module in enumerate(modules):
            if hasattr(module, 'weights_regularizer'):
                # find first subsequent module, from current,
                # with a weight attribute
                for next_module in modules[i:]:
                    if isinstance(next_module, nn.Linear)\
                            or isinstance(next_module, nn.modules.conv._ConvNd):
                        reg_loss += module.weights_regularizer(
                            next_module.weight)
                        if hasattr(module, 'bias') and module.bias is not None:
                            reg_loss += module.biases_regularizer(
                                next_module.bias)
        return reg_loss
