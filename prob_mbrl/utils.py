import numpy as np
import torch

from collections import Iterable
from itertools import chain
from matplotlib import pyplot as plt

from algorithms.mc_pilco import rollout


def iterate_minibatches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)
    N = len(inputs)
    indices = np.arange(0, N)
    np.random.shuffle(indices)
    while True:
        for i in range(0, len(inputs), batchsize):
            idx = indices[i:i + batchsize]
            yield inputs[idx], targets[idx]


def to_complex(x, dims=[]):
    if len(dims) == 0:
        return x
    else:
        dims = dims.to(x.device)
        angles = x.index_select(-1, dims)
        odims = torch.range(0, x.shape[-1]-1).long().to(dims.device)
        odims = (
            1 - torch.eq(dims, odims[:, None])).prod(1).nonzero()[:, 0]
        others = x.index_select(-1, odims)
        return torch.cat([others, angles.sin(), angles.cos()], -1)


def plot_rollout(x0, forward, pol, steps):
    trajs = rollout(
        x0, forward, pol, steps, resample_model=False,
        resample_policy=False, resample_particles=False)
    states, actions, rewards = (
        torch.stack(x).transpose(0,1).cpu().detach().numpy() for x in zip(*trajs))
    names = ['Rolled out States', 'Predicted Actions', 'Predicted Rewards']
    for name in names:
        fig = plt.figure(name)
        fig.clear()

    fig1, axarr1 = plt.subplots(
        states.shape[-1], num=names[0], sharex=True, figsize=(16,9))
    fig2, axarr2 = plt.subplots(
        actions.shape[-1], num=names[1], sharex=True, figsize=(16,3))
    fig3, axarr3 = plt.subplots(
        actions.shape[-1], num=names[2], sharex=True, figsize=(16,3))

    axarr1 = [axarr1] if not isinstance(axarr1, Iterable) else axarr1
    axarr2 = [axarr2] if not isinstance(axarr2, Iterable) else axarr2
    axarr3 = [axarr3] if not isinstance(axarr3, Iterable) else axarr3
    
    for i, (st, ac, rw) in enumerate(zip(states, actions, rewards)):
        H, D = st.shape
        for d in range(D):
            axarr1[d].plot(
                np.arange(H), st[:, d], label='state(%d,%d)' % (d, i),
                color='steelblue', alpha=0.3)

        H, A = ac.shape
        for a in range(A):
            axarr2[a].plot(
                np.arange(H), ac[:, a], label='action(%d,%d)' % (a, i),
                color='steelblue', alpha=0.3)          

        H, R = rw.shape
        for r in range(R):
            axarr3[r].plot(
                np.arange(H), rw[:, r], label='reward(%d,%d)' % (r, i),
                color='steelblue', alpha=0.3)

    for ax in chain(axarr1, axarr2, axarr3):
        ax.figure.canvas.draw()

    plt.show(False)
    plt.waitforbuttonpress(0.5)


def batch_jacobian(f, x, out_dims=None):
    if out_dims is None:
        y = f(x)
        out_dims = y.shape[-1]
    x_rep = x.repeat(out_dims, 1)
    x_rep = torch.tensor(x_rep, requires_grad=True)
    y_rep = f(x_rep)
    dydx = torch.autograd.grad(
        y_rep, x_rep, torch.eye(x.shape[-1]),
        allow_unused=True, retain_graph=True)
    return dydx
