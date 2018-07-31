import numpy as np
import torch

from torch import nn
from torch.nn import Parameter

from ..utils import to_complex


class StochasticModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(StochasticModule, self).__init__(*args, **kwargs)


class BDropout(StochasticModule):
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
        self.register_buffer(
            'regularizer_scale', torch.tensor(0.5*regularizer_scale**2))
        self.register_buffer('rate', torch.tensor(rate))
        self.p = 1 - self.rate
        self.register_buffer('noise', torch.bernoulli(1.0 - self.rate))

    def weights_regularizer(self, weights):
        self.p = 1 - self.rate
        return self.regularizer_scale*(self.p*(weights**2)).sum()

    def biases_regularizer(self, biases):
        return self.regularizer_scale*(biases**2).sum()

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
            return x*torch.bernoulli(self.p.expand(x.shape))
        return x*self.noise

    def extra_repr(self):
        return 'rate={}, regularizer_scale={}'.format(
            self.rate, self.regularizer_scale
        )


class CDropout(BDropout):
    def __init__(self, rate=0.5,
                 name=None,
                 regularizer_scale=1.0,
                 temperature=0.1,
                 **kwargs):
        super(CDropout, self).__init__(
            rate, name, regularizer_scale, **kwargs)
        self.register_buffer('temp', torch.tensor(temperature))
        self.logit_p = Parameter(
            -torch.log(1.0/torch.tensor(1 - self.rate) - 1.0))

    def weights_regularizer(self, weights):
        p = self.logit_p.sigmoid()
        reg = self.regularizer_scale*(p*(weights**2)).sum()
        reg -= -p*p.log() - (1-p)*(1-p).log()
        return reg

    def update_noise(self, x):
        self.noise.data = torch.rand_like(x)

    def forward(self, x, resample=True, mask_dims=2, **kwargs):
        sample_shape = x.shape[-mask_dims:]
        noise = self.noise
        if sample_shape != self.noise.shape:
            sample = x.view(-1, *sample_shape)[0]
            self.update_noise(sample)
            noise = self.noise
        elif resample:
            noise = torch.rand_like(x)

        concrete_p = self.logit_p + noise.log() - (1 - noise).log()
        concrete_noise = (concrete_p/self.temp).sigmoid()

        return x*concrete_noise

    def extra_repr(self):
        return 'rate={}, temperature={}, regularizer_scale={}'.format(
            1-self.logit_p.sigmoid(), self.temp, self.regularizer_scale
        )


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
                    input, resample=resample, repeat_mask=repeat_mask,
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
                                and next_module.bias is not None:
                            reg_loss += module.biases_regularizer(
                                next_module.bias)
                        break
        return reg_loss


class Regressor(torch.nn.Module):
    def __init__(self, model, output_density=None, angle_dims=[]):
        super(Regressor, self).__init__()
        self.model = model
        self.output_density = output_density
        self.register_buffer('angle_dims', torch.tensor(angle_dims).long())
        self.register_buffer('X', torch.ones([1, 1]))
        self.register_buffer('Y', torch.ones([1, 1]))
        self.register_buffer('mx', torch.ones([1, 1]))
        self.register_buffer('Sx', torch.ones([1, 1]))
        self.register_buffer('iSx', torch.ones([1, 1]))
        self.register_buffer('my', torch.ones([1, 1]))
        self.register_buffer('Sy', torch.ones([1, 1]))
        self.register_buffer('iSy', torch.ones([1, 1]))

    def set_dataset(self, X, Y):
        self.X.data = to_complex(X, self.angle_dims)
        self.Y.data = Y
        self.mx.data = self.X.mean(0, keepdim=True)
        self.Sx.data = self.X.std(0, keepdim=True)
        self.iSx.data = self.X.std(0, keepdim=True).reciprocal()
        self.my.data = self.Y.mean(0, keepdim=True)
        self.Sy.data = self.Y.std(0, keepdim=True)
        self.iSy.data = self.Y.std(0, keepdim=True).reciprocal()

    def regularization_loss(self):
        return self.model.regularization_loss()

    def resample(self, *args, **kwargs):
        self.model.resample(*args, **kwargs)
        if self.output_density is not None:
            self.output_density.resample(*args, **kwargs)

    def forward(self, x, normalize=True, **kwargs):
        ''' This assumes that the newtork outputs the parameters for
            an isotropic Gaussian predictive distribution, for each
            batch sample'''
        if x.shape[-1] != self.X.shape[-1]:
            x = to_complex(x, self.angle_dims)
        # scale and center inputs
        if normalize:
            x = (x - self.mx)*self.iSx
        outs = self.model(x, **kwargs)
        if callable(self.output_density):
            scaling_params = (self.my, self.Sy) if normalize else None
            outs = self.output_density(
                outs, scaling_params=scaling_params, **kwargs)

        return outs


class Policy(torch.nn.Module):
    def __init__(self, model, out_scale=1.0, out_bias=0.0, angle_dims=[]):
        super(Policy, self).__init__()
        self.model = model
        self.register_buffer('angle_dims', torch.tensor(angle_dims).long())
        self.register_buffer('scale', torch.tensor(out_scale))
        self.register_buffer('bias', torch.tensor(out_bias))

    def resample(self, *args, **kwargs):
        self.model.resample(*args, **kwargs)

    def forward(self, x, **kwargs):
        return_numpy = isinstance(x, np.ndarray)
        kwargs['resample'] = kwargs.get('resample', True)
        if return_numpy:
            x = torch.tensor(
                x, dtype=self.scale.dtype, device=self.scale.device)
        x = to_complex(x, self.angle_dims)
        u = self.scale*self.model(x, **kwargs) + self.bias
        if return_numpy:
            return u.detach().cpu().numpy()
        else:
            return u


class DynamicsModel(Regressor):
    def __init__(self, model, reward_func=None, **kwargs):
        super(DynamicsModel, self).__init__(model, **kwargs)
        self.register_buffer('maxR', torch.ones([1, 1]))
        self.register_buffer('minR', torch.ones([1, 1]))
        self.reward_func = reward_func

    def set_dataset(self, X, Y):
        super(DynamicsModel, self).set_dataset(X, Y)
        D = self.Y.shape[-1]-1
        R = self.Y.index_select(-1, torch.tensor(D, device=self.Y.device))
        self.maxR.data = R.max()
        self.minR.data = R.min()

    def forward(self, inputs, separate_outputs=False, **kwargs):
        inputs_as_tuple = isinstance(inputs, tuple) or isinstance(inputs, list)
        if inputs_as_tuple:
            prev_states, actions = inputs[0], inputs[1]
            inputs = torch.cat([prev_states, actions], -1)

        # forward pass on model
        outs = super(DynamicsModel, self).forward(
            inputs, **kwargs)
        if isinstance(outs, tuple):
            return outs
        if callable(self.reward_func):
            # if we have a known reward function
            states = outs
            if not inputs_as_tuple:
                D = outs.shape[-1]
                prev_states, actions = inputs.split(D, -1)
            rewards = self.reward_func(prev_states)
        else:
            D = outs.shape[-1] - 1
            # assume rewards come from the last dimension of the output
            states, rewards = outs.split(D, -1)
            # constrain rewards
            # rewards = (self.maxR - self.minR)*rewards.sigmoid() + self.minR
        if separate_outputs:
            return states, rewards
        return torch.cat([states, rewards], -1)
