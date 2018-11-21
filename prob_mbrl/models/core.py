import inspect
import torch
from .modules import BDropout, BSequential
from collections import OrderedDict, Iterable
from functools import partial


def dropout_mlp(input_dims,
                output_dims,
                hidden_dims=[200, 200],
                nonlin=torch.nn.ReLU,
                output_nonlin=None,
                weights_initializer=partial(
                    torch.nn.init.xavier_normal_,
                    gain=torch.nn.init.calculate_gain('relu')),
                biases_initializer=partial(
                    torch.nn.init.uniform_, a=-0.01, b=0.01),
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
