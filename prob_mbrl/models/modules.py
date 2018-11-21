import torch

from torch import nn
from torch.nn import Parameter


class StochasticModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(StochasticModule, self).__init__(*args, **kwargs)


class BDropout(StochasticModule):
    """
        Extends the base Dropout layer by adding a regularizer as derived by
        Gal and Ghahrahmani "Dropout as a Bayesian Approximation" (2015)
    """

    def __init__(self, rate=0.5, name=None, regularizer_scale=1.0, **kwargs):
        super(BDropout, self).__init__(**kwargs)
        self.name = name
        self.register_buffer('regularizer_scale',
                             torch.tensor(0.5 * regularizer_scale**2))
        self.register_buffer('rate', torch.tensor(rate))
        self.p = 1 - self.rate
        self.register_buffer('noise', torch.bernoulli(1.0 - self.rate))

    def weights_regularizer(self, weights):
        self.p = 1 - self.rate
        return self.regularizer_scale * (self.p * (weights**2)).sum()

    def biases_regularizer(self, biases):
        return self.regularizer_scale * (biases**2).sum()

    def resample(self):
        self.update_noise(self.noise)

    def update_noise(self, x):
        self.p = 1 - self.rate
        self.noise.data = torch.bernoulli(self.p.expand(x.shape))

    def forward(self, x, resample=True, mask_dims=2, **kwargs):
        sample_shape = x.shape[-mask_dims:]
        if sample_shape != self.noise.shape:
            sample = x.view(-1, *sample_shape)[0]
            self.update_noise(sample)
        elif resample:
            return x * torch.bernoulli(self.p.expand(x.shape))

        # we never need the noise gradients
        return x * self.noise.detach()

    def extra_repr(self):
        return 'rate={}, regularizer_scale={}'.format(self.rate,
                                                      self.regularizer_scale)


class CDropout(BDropout):
    def __init__(self,
                 rate=0.5,
                 name=None,
                 regularizer_scale=1.0,
                 temperature=0.1,
                 **kwargs):
        super(CDropout, self).__init__(rate, name, regularizer_scale, **kwargs)
        self.register_buffer('temp', torch.tensor(temperature))
        self.logit_p = Parameter(
            -torch.log(1.0 / torch.tensor(1 - self.rate) - 1.0))
        self.concrete_noise = None

    def weights_regularizer(self, weights):
        p = self.logit_p.sigmoid()
        reg = self.regularizer_scale * (p * (weights**2)).sum()
        reg -= -p * p.log() - (1 - p) * (1 - p).log()
        return reg

    def update_noise(self, x):
        self.noise.data = torch.rand_like(x)

    def update_concrete_noise(self, noise):
        """Updates the concrete dropout masks.

        Args:
            noise (Tensor): Input.
        """
        self.p.data = self.logit_p.sigmoid()
        concrete_p = self.logit_p + noise.log() - (1 - noise).log()
        self.concrete_noise = (concrete_p / self.temp).sigmoid()

    def forward(self, x, resample=False, mask_dims=2, **kwargs):
        """Computes the concrete dropout.

        Args:
            x (Tensor): Input.
            resample (bool): Whether to force resample.
            mask_dims (int): Number of dimensions to sample noise for
                (0 for all).

        Returns:
            Output (Tensor).
        """
        sample_shape = x.shape[-mask_dims:]
        noise = self.noise
        resampled = False
        if resample:
            noise = torch.rand_like(x)
            resampled = True
        elif (self.concrete_noise is None
              or sample_shape != self.concrete_noise.shape):
            sample = x.view(-1, *sample_shape)[0]
            self.update_noise(sample)
            noise = self.noise
            resampled = True

        if self.training:
            self.update_concrete_noise(noise)
            concrete_noise = self.concrete_noise
        else:
            if resampled:
                self.update_concrete_noise(noise)
            # We never need these gradients in evaluation mode.
            concrete_noise = self.concrete_noise.detach()
        return x * concrete_noise

    def extra_repr(self):
        return 'rate={}, temperature={}, regularizer_scale={}'.format(
            1 - self.logit_p.sigmoid(), self.temp, self.regularizer_scale)


class TLNDropout(BDropout):
    '''
        'Implements truncated log-normal dropout (NIPS 2017)
    '''

    def __init__(self, interval=[-10, 0]):
        self.register_buffer('interval', torch.tensor(interval))
        self.logit_posterior_mean = Parameter(
            -torch.log(1.0 / torch.tensor(1 - self.rate) - 1.0))
        # self.logit_posterior_std = logit_posterior_std

    def weights_regularizer(self, weights):
        '''
        In this case the weights regularizer is actually independent of the
        weights (only depends on the alpha parameter)
        '''
        return 0

    def update_noise(self, x):
        pass

    def forward(self, x):
        pass


class BSequential(nn.modules.Sequential):
    " An extension to sequential that allows for controlling resampling"

    def __init__(self, *args):
        super(BSequential, self).__init__(*args)

    def resample(self):
        for module in self._modules.values():
            if isinstance(module, BDropout):
                module.resample()

    def forward(self, input, resample=True, repeat_mask=False, **kwargs):
        for module in self._modules.values():
            if isinstance(module, StochasticModule):
                input = module(
                    input,
                    resample=resample,
                    repeat_mask=repeat_mask,
                    **kwargs)
            else:
                input = module(input)
        return input

    def regularization_loss(self):
        modules = self._modules.values()
        reg_loss = 0
        for i, module in enumerate(modules):
            if hasattr(module, 'weights_regularizer'):
                # find first subsequent module, from current,output_samples
                # with a weight attribute
                for next_module in modules[i:]:
                    if isinstance(next_module, nn.Linear)\
                            or isinstance(next_module,
                                          nn.modules.conv._ConvNd):
                        reg_loss += module.weights_regularizer(
                            next_module.weight)
                        if hasattr(next_module, 'bias')\
                                and next_module.bias is not None\
                                and hasattr(module, 'biases_regularizer'):
                            reg_loss += module.biases_regularizer(
                                next_module.bias)
                        break
        return reg_loss
