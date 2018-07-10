import numpy as np
import torch
import warnings

from torch import nn
from torch.nn import Parameter

from ..utils import to_complex


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

    def forward(self, x, resample=True, repeat_mask=False):
        if (x.shape != self.noise.shape and not repeat_mask) or resample:
            self.update_noise(x)
        if repeat_mask:
            return x*self.noise.repeat(repeat_mask, 1)
        else:
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

    def forward(self, x, resample=True, repeat_mask=False):
        if (x.shape != self.noise.shape and not repeat_mask) or resample:
            self.update_noise(x)

        p = self.logit_p.sigmoid()
        concrete_p = p.log() - (1-p).log()\
            + self.noise.log() - (1 - self.noise).log()
        concrete_noise = (concrete_p/self.temp).sigmoid()

        if repeat_mask:
            return x*concrete_noise.repeat(repeat_mask, 1)
        else:
            return x*concrete_noise


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
            if isinstance(module, BDropout):
                input = module(
                    input, resample=resample, repeat_mask=repeat_mask)
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
                            or isinstance(next_module,
                                          nn.modules.conv._ConvNd):
                        reg_loss += module.weights_regularizer(
                            next_module.weight)
                        if hasattr(module, 'bias') and module.bias is not None:
                            reg_loss += module.biases_regularizer(
                                next_module.bias)
        return reg_loss


class DiagGaussianDensity(torch.nn.Module):
    '''
        Rearranges the incoming dimensions to correspond to the parameters
        of a Gaussian Density distribution.
    '''
    def __init__(self, output_dims):
        super(DiagGaussianDensity, self).__init__()
        self.output_dims = output_dims

    def forward(self, x, scaling_params=None, *args, **kwargs):
        D = self.output_dims
        idx = torch.range(0, 2*D-1, dtype=torch.long, device=x.device)
        mean = x.index_select(-1, idx[:D])
        std = x.index_select(-1, idx[D:]).sigmoid()

        # scale and center outputs
        if scaling_params is not None and len(scaling_params) > 0:
            if len(scaling_params) == 2:
                my = scaling_params[0]
                Sy = scaling_params[1]
                mean *= Sy
                std *= Sy
                mean += my
            else:
                warnings.warn(
                    "Expected scaling_params as tuple or list with 2 elements")
        return mean, std


class MixtureDensity(torch.nn.Module):
    '''
     Mixture of Gaussians density Network model. The components have diagonal
     covariance.
    '''
    def __init__(self, output_dims, n_components, **kwargs):
        super(MixtureDensity, self).__init__(**kwargs)
        self.n_components = n_components
        self.output_dims = output_dims

    def forward(self, x, scaling_params=None, *args, **kwargs):
        D = self.output_dims
        nD = D*self.n_components

        idx = torch.range(
            0, 2*nD+self.n_components-1, dtype=torch.long, device=x.device)
        mean = x.index_select(-1, idx[:nD]).view(-1, self.n_components, D)
        std = x.index_select(
            -1, idx[nD:2*nD]).view(-1, self.n_components, D).sigmoid()
        if self.n_components > 1:
            logit_pi = x.index_select(-1, idx[2*nD:])
            pi = torch.nn.functional.softmax(logit_pi, -1)
        else:
            pi = 1

        # scale and center outputs
        if scaling_params is not None and len(scaling_params) > 0:
            if len(scaling_params) == 2:
                my = scaling_params[0]
                Sy = scaling_params[1]
                mean *= Sy
                std *= Sy
                mean += my
            else:
                warnings.warn(
                    "Expected scaling_params as tuple or list with 2 elements")
        return mean, std, pi


class Regressor(torch.nn.Module):
    def __init__(self, model, output_density=None, angle_dims=[]):
        super(Regressor, self).__init__()
        self.model = model
        self.output_density = output_density
        self.angle_dims = torch.nn.Parameter(
            torch.tensor(angle_dims).long(), requires_grad=False)
        self.X = torch.nn.Parameter(torch.empty([1, 1]), requires_grad=False)
        self.Y = torch.nn.Parameter(torch.empty([1, 1]), requires_grad=False)
        self.mx = torch.nn.Parameter(torch.empty([1, 1]), requires_grad=False)
        self.Sx = torch.nn.Parameter(torch.empty([1, 1]), requires_grad=False)
        self.iSx = torch.nn.Parameter(torch.empty([1, 1]), requires_grad=False)
        self.my = torch.nn.Parameter(torch.empty([1, 1]), requires_grad=False)
        self.Sy = torch.nn.Parameter(torch.empty([1, 1]), requires_grad=False)
        self.iSy = torch.nn.Parameter(torch.empty([1, 1]), requires_grad=False)

    def set_dataset(self, X, Y):
        self.X.data = to_complex(X, self.angle_dims)
        self.Y.data = Y
        self.mx.data = self.X.mean(0)
        self.Sx.data = self.X.std(0)
        self.iSx.data = self.X.std(0).reciprocal()
        self.my.data = self.Y.mean(0)
        self.Sy.data = self.Y.std(0)
        self.iSy.data = self.Y.std(0).reciprocal()

    def regularization_loss(self):
        return self.model.regularization_loss()

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
            outs = self.output_density(outs, scaling_params=scaling_params)

        return outs


class Policy(torch.nn.Module):
    def __init__(self, model, out_scale=1.0, out_bias=0.0, angle_dims=[]):
        super(Policy, self).__init__()
        self.model = model
        self.angle_dims = torch.nn.Parameter(
            torch.tensor(angle_dims).long(), requires_grad=False)
        self.scale = torch.nn.Parameter(
            torch.tensor(out_scale), requires_grad=False)
        self.bias = torch.nn.Parameter(
            torch.tensor(out_bias), requires_grad=False)

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
    def __init__(self, model, reward_func=None, angle_dims=[]):
        super(DynamicsModel, self).__init__(model, angle_dims)
        self.maxR = torch.nn.Parameter(
            torch.empty([1, 1]), requires_grad=False)
        self.minR = torch.nn.Parameter(
            torch.empty([1, 1]), requires_grad=False)
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
        pred_means, pred_stds = super(DynamicsModel, self).forward(
            inputs, **kwargs)

        if callable(self.reward_func):
            # if we have a known reward function
            states, states_std = pred_means, pred_stds
            if not inputs_as_tuple:
                D = pred_means.shape[-1]
                prev_states = inputs.index_select(
                    -1, torch.range(0, D-1, device=inputs.device).long())
            rewards, rewards_std = self.reward_func(prev_states)
            if separate_outputs:
                return (states, states_std), (rewards, rewards_std)
            else:
                return states, states_std
        else:
            D = pred_means.shape[-1] - 1
            # assume loss comes from the last dimension of the output
            states = pred_means.index_select(
                -1, torch.range(0, D-1, device=inputs.device).long())
            rewards = pred_means.index_select(
                -1, torch.tensor(D, device=inputs.device))
            # constrain rewards
            # rewards = (self.maxR - self.minR)*rewards.sigmoid() + self.minR
            if separate_outputs:
                states_std = pred_stds.index_select(
                    -1, torch.range(0, D-1, device=inputs.device).long())
                rewards_std = pred_stds.index_select(
                    -1, torch.tensor(D, device=inputs.device))
                return (states, states_std), (rewards, rewards_std)

        return torch.cat([states, rewards], -1), pred_stds
