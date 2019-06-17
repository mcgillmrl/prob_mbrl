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
        self.logit_p = Parameter(-torch.log(1.0 / (1 - self.rate) - 1.0))
        self.register_buffer('concrete_noise',
                             torch.bernoulli(1.0 - self.rate))

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
        elif (sample_shape != self.concrete_noise.shape):
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
        modules = list(self._modules.values())
        for i, module in enumerate(modules):
            if isinstance(module, BDropout) or isinstance(
                    module, torch.nn.Dropout):
                if i < len(modules):
                    next_module = modules[i + 1]
                    if isinstance(next_module, SpectralNorm):
                        # rescale lipschitz constant by dropout probability
                        pass
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
        modules = list(self._modules.values())
        reg_loss = 0
        for i, module in enumerate(modules):
            if hasattr(module, 'weights_regularizer'):
                # find first subsequent module, from current,output_samples
                # with a weight attribute
                for next_module in modules[i:]:
                    if isinstance(next_module, SpectralNorm):
                        next_module = next_module.module
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
            elif hasattr(module, 'regularization_loss'):
                reg_loss += module.regularization_loss()
        return reg_loss


class SpectralNorm(torch.nn.Module):
    """
        Applies spectral normalization to the weights matrix, i.e.
        W_sn = W/sigma(W), where sigma(W) is the largest eigenvalue
        of W
    """

    def __init__(self,
                 module,
                 power_iterations=1,
                 max_K=10,
                 param_name='weight',
                 train_scale=False):
        assert hasattr(module, param_name)
        super(SpectralNorm, self).__init__()
        self.module = module
        self.param_name = param_name
        self.n_iter = power_iterations
        self.max_K = max_K
        self.init_params()
        self.scale = torch.nn.Parameter(
            torch.zeros(1), requires_grad=train_scale)

    def extra_repr(self):
        if hasattr(self.module, self.param_name):
            w = getattr(self.module, self.param_name)
        else:
            w = getattr(self.module, self.param_name + '_bar')
        return "scale={}, norm={}".format(
            (self.max_K * self.scale.sigmoid()).data,
            torch.svd(w)[1][0])

    def init_params(self):
        w = self.module._parameters[self.param_name]
        w_sn = torch.nn.Parameter(w.data)
        M = w.shape[0]
        N = w.view(M, -1).shape[0]
        u = torch.randn(M).to(w.device, w.dtype)
        u.data = u / (u.norm() + 1e-12)
        v = torch.randn(N).to(w.device, w.dtype)
        v.data = v / (v.norm() + 1e-12)

        self.module.register_parameter(self.param_name + "_bar", w_sn)
        self.module.register_buffer(self.param_name + "_u", u)
        self.module.register_buffer(self.param_name + "_v", v)

        del self.module._parameters[self.param_name]

    def power_iteration(self, n_iters=1):
        u = getattr(self.module, self.param_name + '_u')
        v = getattr(self.module, self.param_name + '_v')
        w = getattr(self.module, self.param_name + '_bar')
        M = w.shape[0]
        w_square = w.view(M, -1)

        for i in range(n_iters):
            v_ = torch.mv(w_square.data.transpose(0, 1), u.data)
            v.data = v_ / (v_.norm() + 1e-12)
            u_ = torch.mv(w_square.data, v.data)
            u.data = u_ / (u_.norm() + 1e-12)

        sigma_w = u.dot(w.view(M, -1).mv(v))
        setattr(self.module, self.param_name,
                self.max_K * self.scale.sigmoid() * (w / sigma_w.expand_as(w)))

    def forward(self, *args, **kwargs):
        if self.training:
            self.power_iteration(self.n_iter)
        return self.module(*args, **kwargs)
