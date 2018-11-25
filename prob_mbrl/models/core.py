import inspect
import numpy as np
import torch
from .modules import BDropout, BSequential
from collections import OrderedDict, Iterable
from functools import partial

from ..utils.angles import to_complex


def mlp(input_dims,
        output_dims,
        hidden_dims=[200, 200],
        nonlin=torch.nn.ReLU,
        output_nonlin=None,
        weights_initializer=partial(
            torch.nn.init.xavier_normal_,
            gain=torch.nn.init.calculate_gain('relu')),
        biases_initializer=partial(torch.nn.init.uniform_, a=-0.01, b=0.01),
        dropout_layers=BDropout,
        input_dropout=None):
    '''
        Utility function for creating multilayer perceptrons of varying depth.
    '''
    dims = [input_dims] + hidden_dims
    if not isinstance(dropout_layers, Iterable):
        dropout_layers = [dropout_layers] * (len(hidden_dims))

    modules = OrderedDict()
    # add input dropout
    if inspect.isclass(input_dropout):
        input_dropout = input_dropout(name='drop_input')
    if input_dropout is not None:
        modules['drop_input'] = input_dropout

    # add hidden layers
    for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        drop_i = dropout_layers[i]
        if inspect.isclass(drop_i):
            drop_i = drop_i(name='drop%d' % i)
        # fully connected layer
        modules['fc%d' % i] = torch.nn.Linear(din, dout)
        # activation
        modules['nonlin%d' % i] = nonlin()
        # dropout (regularizes next layer)
        if drop_i is not None:
            modules['drop%d' % i] = drop_i

    # project to output dimensions
    modules['fc_out'] = torch.nn.Linear(dims[-1], output_dims)
    # add output activation, if specified
    if output_nonlin is not None:
        modules['fc_nonlin'] = output_nonlin()

    # build module
    net = BSequential(modules)

    # initialize weights
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
    def __init__(self, model, output_density=None, angle_dims=[]):
        super(Regressor, self).__init__()
        self.model = model
        self.output_density = output_density
        self.register_buffer('angle_dims', torch.tensor(angle_dims).long())
        self.register_buffer('X', torch.ones([1, 1]))
        self.register_buffer('Y', torch.ones([1, 1]))
        self.register_buffer('mx', torch.zeros([1, 1]))
        self.register_buffer('Sx', torch.ones([1, 1]))
        self.register_buffer('iSx', torch.ones([1, 1]))
        self.register_buffer('my', torch.zeros([1, 1]))
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
            x = (x - self.mx) * self.iSx
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
        else:
            x = x.to(self.scale.device)
        x = to_complex(x, self.angle_dims)
        u = self.scale * self.model(x, **kwargs) + self.bias
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
        D = self.Y.shape[-1] - 1
        R = self.Y.index_select(-1, torch.tensor(D, device=self.Y.device))
        self.maxR.data = R.max()
        self.minR.data = R.min()

    def forward(self, inputs, separate_outputs=False, deltas=True, **kwargs):
        inputs_as_tuple = isinstance(inputs, tuple) or isinstance(inputs, list)
        if inputs_as_tuple:
            prev_states, actions = inputs[0], inputs[1]
            inputs = torch.cat([prev_states, actions], -1)

        # forward pass on model
        outs = super(DynamicsModel, self).forward(inputs, **kwargs)

        # if not returning samples, outs will be a tuple consisting of the
        # parameters of the output distribution
        if isinstance(outs, tuple):
            return outs

        # if we are returning samples, the output will be a tensor whose last
        # dimension is of size D+1, when the reward function is being learned,
        # or of size D, when an external reward function is available.
        if not inputs_as_tuple:
            D = outs.shape[-1]
            prev_states, actions = inputs.split(D, -1)

        if callable(self.reward_func):
            # if we have a known reward function
            dstates = outs
            rewards = self.reward_func(prev_states + dstates, actions)
        else:
            D = outs.shape[-1] - 1
            # assume rewards come from the last dimension of the output
            dstates, rewards = outs.split(D, -1)

        states = dstates if deltas else prev_states + dstates

        if separate_outputs:
            return states, rewards

        return torch.cat([states, rewards], -1)