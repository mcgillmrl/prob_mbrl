import inspect
import numpy as np
import torch
from .modules import BDropout, BSequential
from collections import OrderedDict, Iterable
from functools import partial

from ..utils import to_complex


def mlp(input_dims, output_dims, hidden_dims=[200, 200], nonlin=torch.nn.ReLU,
        weights_initializer=partial(torch.nn.init.xavier_normal_,
                                    gain=torch.nn.init.calculate_gain('relu')),
        biases_initializer=partial(torch.nn.init.uniform_, a=-0.1, b=0.1)):
    dims = [input_dims]+hidden_dims
    modules = OrderedDict()
    for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        modules['fc%d' % i] = torch.nn.Linear(din, dout)
        modules['nonlin%d' % i] = nonlin()
    modules['fc_out'] = torch.nn.Linear(dims[-1], output_dims)
    net = BSequential(modules)
    if callable(weights_initializer):
        def fn(module):
            if hasattr(module, 'weight'):
                weights_initializer(module.weight)
        net.apply(fn)
    if callable(biases_initializer):
        def fn(module):
            if hasattr(module, 'bias'):
                biases_initializer(module.bias)
        net.apply(fn)
    return net


def dropout_mlp(input_dims, output_dims, hidden_dims=[200, 200],
                nonlin=torch.nn.ReLU,
                output_nonlin=None,
                weights_initializer=partial(
                    torch.nn.init.xavier_normal_,
                    gain=torch.nn.init.calculate_gain('relu')),
                biases_initializer=partial(
                    torch.nn.init.uniform_, a=-1.0, b=1.0),
                dropout_layers=BDropout):
    dims = [input_dims]+hidden_dims
    if not isinstance(dropout_layers, Iterable):
        dropout_layers = [dropout_layers]*(len(hidden_dims))

    modules = OrderedDict()
    for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        drop_i = dropout_layers[i]
        if inspect.isclass(drop_i):
            drop_i = drop_i(name='drop%d' % i)

        modules['fc%d' % i] = torch.nn.Linear(din, dout)
        modules['drop%d' % i] = drop_i
        modules['nonlin%d' % i] = nonlin()
    modules['fc_out'] = torch.nn.Linear(dims[-1], output_dims)
    if output_nonlin is not None:
        modules['fc_nonlin'] = output_nonlin()

    net = BSequential(modules)
    if callable(weights_initializer):
        def fn(module):
            if hasattr(module, 'weight'):
                weights_initializer(module.weight)
        net.apply(fn)
    if callable(biases_initializer):
        def fn(module):
            if hasattr(module, 'bias') and module.bias is not None:
                biases_initializer(module.bias)
        net.apply(fn)
    return net


class Regressor(torch.nn.Module):
    def __init__(self, model):
        super(Regressor, self).__init__()
        self.model = model
        self.X = torch.nn.Parameter(torch.empty([1, 1]), requires_grad=False)
        self.Y = torch.nn.Parameter(torch.empty([1, 1]), requires_grad=False)
        self.mx = torch.nn.Parameter(torch.empty([1, 1]), requires_grad=False)
        self.Sx = torch.nn.Parameter(torch.empty([1, 1]), requires_grad=False)
        self.my = torch.nn.Parameter(torch.empty([1, 1]), requires_grad=False)
        self.Sy = torch.nn.Parameter(torch.empty([1, 1]), requires_grad=False)

    def set_dataset(self, X, Y):
        self.X.data = X
        self.Y.data = Y
        self.mx.data = self.X.mean(0)
        self.Sx.data = self.X.std(0)
        self.my.data = self.Y.mean(0)
        self.Sy.data = self.Y.std(0)

    def regularization_loss(self):
        return self.model.regularization_loss()

    def forward(self, x, normalize=True, **kwargs):
        ''' This assumes that the newtork outputs the parameters for
            an isotropic Gaussian predictive distribution, for each
            batch sample'''
        # scale and center inputs
        if normalize:
            x = (x - self.mx)/self.Sx
        outs = self.model(x, **kwargs)

        # scale and center outputs
        D = outs.shape[-1]/2
        idx = torch.range(0, 2*D-1, dtype=torch.long, device=x.device)
        pred_mean = outs.index_select(-1, idx[:D])
        pred_std = outs.sigmoid().index_select(-1, idx[D:])
        if normalize:
            pred_mean *= self.Sy
            pred_std *= self.Sy
            pred_mean += self.my
        return pred_mean, pred_std


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
        super(DynamicsModel, self).__init__(model)
        self.angle_dims = torch.nn.Parameter(
            torch.tensor(angle_dims).long(), requires_grad=False)
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
