import numpy as np
import torch

from collections import Iterable
from itertools import chain
from matplotlib import pyplot as plt

from ..algorithms.mc_pilco import rollout


def iterate_minibatches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)
    N = len(inputs)
    indices = np.arange(0, N)
    np.random.shuffle(indices)
    while True:
        for i in range(0, len(inputs), batchsize):
            idx = indices[i:i + batchsize]
            yield inputs[idx], targets[idx]


def plot_sample(data, axarr, colors=None, **kwargs):
    H, D = data.shape
    plots = []
    if colors is None:
        colors = ['steelblue'] * D
    N = len(colors)
    for d in range(D):
        pl, = axarr[d].plot(
            np.arange(H), data[:, d], color=colors[d % N], **kwargs)
        plots.append(pl)
    return plots


def plot_mean_var(data, axarr, colors=None, stdevs=2, **kwargs):
    N, H, D = data.shape
    plots = []
    mean = data.mean(0)
    std = data.std(0)
    t = np.arange(H)
    if colors is None:
        colors = ['steelblue'] * D
    N = len(colors)
    for d in range(D):
        pl, = axarr[d].plot(t, mean[:, d], color=colors[d % N], **kwargs)
        alpha = kwargs.get('alpha', 0.5)
        for i in range(1, stdevs + 1):
            alpha = alpha * 0.8
            lower_bound = mean[:, d] - i * std[:, d]
            upper_bound = mean[:, d] + i * std[:, d]
            axarr[d].fill_between(
                t, lower_bound, upper_bound, alpha=alpha, color=pl.get_color())
        plots.append(pl)
    return plots


def plot_trajectories(
        states,
        actions,
        rewards,
        names=['Rolled out States', 'Predicted Actions', 'Predicted Rewards'],
        timeout=0.5,
        plot_samples=True):
    for name in names:
        fig = plt.figure(name)
        fig.clear()

    fig1, axarr1 = plt.subplots(
        states.shape[-1], num=names[0], sharex=True, figsize=(16, 9))
    fig2, axarr2 = plt.subplots(
        actions.shape[-1], num=names[1], sharex=True, figsize=(16, 3))
    fig3, axarr3 = plt.subplots(
        actions.shape[-1], num=names[2], sharex=True, figsize=(16, 3))

    axarr1 = [axarr1] if not isinstance(axarr1, Iterable) else axarr1
    axarr2 = [axarr2] if not isinstance(axarr2, Iterable) else axarr2
    axarr3 = [axarr3] if not isinstance(axarr3, Iterable) else axarr3
    if plot_samples:
        c1 = c2 = c3 = None
        for i, (st, ac, rw) in enumerate(zip(states, actions, rewards)):
            r1 = plot_sample(st, axarr1, c1, alpha=0.3)
            r2 = plot_sample(ac, axarr2, c2, alpha=0.3)
            r3 = plot_sample(rw, axarr3, c3, alpha=0.3)
            c1 = [r.get_color() for r in r1]
            c2 = [r.get_color() for r in r2]
            c3 = [r.get_color() for r in r3]

    else:
        plot_mean_var(states, axarr1)
        plot_mean_var(actions, axarr2)
        plot_mean_var(rewards, axarr3)

    for ax in chain(axarr1, axarr2, axarr3):
        ax.figure.canvas.draw()

    plt.show(False)
    plt.waitforbuttonpress(timeout)


def plot_rollout(x0, forward, pol, steps):
    trajs = rollout(
        x0,
        forward,
        pol,
        steps,
        resample_model=False,
        resample_policy=False,
        resample_particles=False)
    states, actions, rewards = (torch.stack(x).transpose(
        0, 1).cpu().detach().numpy() for x in zip(*trajs))

    plot_trajectories(states, actions, rewards)


def batch_jacobian(f, x, out_dims=None):
    if out_dims is None:
        y = f(x)
        out_dims = y.shape[-1]
    x_rep = x.repeat(out_dims, 1)
    x_rep = torch.tensor(x_rep, requires_grad=True)
    y_rep = f(x_rep)
    dydx = torch.autograd.grad(
        y_rep,
        x_rep,
        torch.eye(x.shape[-1]),
        allow_unused=True,
        retain_graph=True)
    return dydx
