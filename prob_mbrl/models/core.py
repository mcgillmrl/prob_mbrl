import copy
import inspect
import multiprocessing
import numpy as np
import torch
from .modules import BDropout, BSequential, SpectralNorm
from collections import OrderedDict, Iterable
from functools import partial

from ..utils.angles import to_complex


def mlp(input_dims,
        output_dims,
        hidden_dims=[200, 200],
        nonlin=torch.nn.ReLU,
        output_nonlin=None,
        weights_initializer=partial(torch.nn.init.xavier_normal_,
                                    gain=torch.nn.init.calculate_gain('relu')),
        biases_initializer=partial(torch.nn.init.uniform_, a=-0.1, b=0.1),
        hidden_biases=True,
        output_biases=True,
        dropout_layers=BDropout,
        input_dropout=None,
        spectral_norm=False,
        spectral_norm_output=False):
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
        fc = torch.nn.Linear(din, dout, bias=hidden_biases)
        if spectral_norm:
            fc = SpectralNorm(fc)
        modules['fc%d' % i] = fc

        # activation
        if callable(nonlin):
            modules['nonlin%d' % i] = nonlin()
        # dropout (regularizes next layer)
        if drop_i is not None:
            modules['drop%d' % i] = drop_i

    # project to output dimensions
    fc_out = torch.nn.Linear(dims[-1], output_dims, bias=output_biases)
    if spectral_norm_output:
        fc_out = SpectralNorm(fc_out)
    modules['fc_out'] = fc_out
    # add output activation, if specified
    if callable(output_nonlin):
        modules['fc_nonlin'] = output_nonlin()

    # build module
    net = BSequential(modules)

    # initialize weights
    def reset_fn():
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

    reset_fn()
    net.float()

    return net


class ModelEnsemble(torch.nn.Module):
    def __init__(self, model, N_ensemble=5):
        super(ModelEnsemble, self).__init__()
        self.N_ensemble = N_ensemble
        for i in range(N_ensemble):
            setattr(self, 'model_%d' % i, copy.deepcopy(model))

    def f(self, args):
        x, i, args, kwargs = args
        model = getattr(self, 'model_%d' % i)
        return model(x, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        pool = multiprocessing.Pool(processes=3)
        ret = pool.map(self.f)
        pool.close
        return ret


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

    def set_dataset(self, X, Y, N_ensemble=-1, p=0.5):
        if len(self.angle_dims):
            self.X.data = to_complex(X, self.angle_dims)
        else:
            self.X.data = X
        self.Y.data = Y
        self.mx.data = self.X.mean(0, keepdim=True)
        self.Sx.data = self.X.std(0, keepdim=True)
        self.Sx.data[self.Sx == 0] = 1.0
        self.iSx.data = self.Sx.reciprocal()
        self.my.data = self.Y.mean(0, keepdim=True)
        self.Sy.data = self.Y.std(0, keepdim=True)
        self.Sy.data[self.Sy == 0] = 1.0
        self.iSy.data = self.Sy.reciprocal()
        if N_ensemble > 1:
            self.masks.data = torch.bernoulli(
                p * torch.ones(X.shape[0], N_ensemble))

    def load(self, state_dict):
        params = dict(self.named_parameters())
        params.update(self.named_buffers())
        for k, v in state_dict.items():
            if k in params:
                params[k].data = v.data.clone()

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
        if len(self.angle_dims) > 0:
            x = to_complex(x, self.angle_dims)
        # scale and center inputs
        if normalize:
            x = (x - self.mx) * self.iSx
        outs = self.model(x, **kwargs)
        if callable(self.output_density):
            scaling_params = (self.my, self.Sy) if normalize else None
            outs = self.output_density(outs,
                                       scaling_params=scaling_params,
                                       **kwargs)

        return outs


class Policy(torch.nn.Module):
    def __init__(
            self,
            model,
            maxU=1.0,
            minU=None,
            angle_dims=[],
    ):
        super(Policy, self).__init__()
        self.model = model
        self.register_buffer('angle_dims', torch.tensor(angle_dims).long())
        if minU is None:
            minU = -maxU
        scale = 0.5 * (maxU - minU)
        bias = 0.5 * (maxU + minU)
        self.register_buffer('scale', torch.tensor(scale).squeeze())
        self.register_buffer('bias', torch.tensor(bias).squeeze())

    def regularization_loss(self):
        return self.model.regularization_loss()

    def resample(self, *args, **kwargs):
        self.model.resample(*args, **kwargs)

    def load(self, state_dict):
        params = dict(self.named_parameters())
        params.update(self.named_buffers())
        for k, v in state_dict.items():
            if k in params:
                params[k].data = v.data.clone()

    def forward(self, x, **kwargs):
        return_numpy = isinstance(x, np.ndarray)
        kwargs['resample'] = kwargs.get('resample', True)
        kwargs['return_samples'] = kwargs.get('return_samples', True)
        if return_numpy:
            x = torch.tensor(x,
                             dtype=self.scale.dtype,
                             device=self.scale.device)
        else:
            x = x.to(dtype=self.scale.dtype, device=self.scale.device)
        if x.dim() == 1:
            x = x[None, :]
        if len(self.angle_dims) > 0:
            x = to_complex(x, self.angle_dims)
        u = self.model(x, **kwargs)

        if isinstance(u, tuple):
            u, unoise = u
            u = u + unoise

        # saturate output
        u = self.scale * u.tanh() + self.bias

        if return_numpy:
            return u.detach().cpu().numpy()
        else:
            return u


class DynamicsModel(Regressor):
    def __init__(self, model, reward_func=None, predict_done=False, **kwargs):
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
        return_samples = kwargs.get("return_samples", False)
        if not return_samples:
            return outs

        # if we are returning samples, the output will be either:
        # 1) a tensor whose last dimension is of size D+1, when the reward
        #    function is being learned, or of size D, when an external reward
        #    function is available.
        # 2) a tuple where the  first element is the tensor described above and
        #    the a corresponding tuple of measurement noise samples
        if not inputs_as_tuple:
            D = outs.shape[-1]
            prev_states, actions = inputs.split(D, -1)

        state_noise = torch.zeros_like(prev_states)
        reward_noise = torch.zeros_like(state_noise[:, 0:1])

        if callable(self.reward_func):
            # if we have a known reward function
            if len(outs) == 2:  # density is returning the output noise
                dstates, state_noise = outs
            else:
                dstates = outs
            rewards = self.reward_func(prev_states + dstates, actions)
        else:
            if len(outs) == 2:  # density is returning the output noise
                outs, noise = outs
                D = outs.shape[-1] - 1
                state_noise, reward_noise = noise.split(D, -1)
            else:
                D = outs.shape[-1] - 1
            # assume rewards come from the last dimension of the output
            dstates, rewards = outs.split(D, -1)

        states = dstates if deltas else prev_states + dstates

        if separate_outputs:
            return (states, rewards), (state_noise, reward_noise)

        return torch.cat([states, rewards],
                         -1), torch.cat([state_noise, reward_noise], -1)
