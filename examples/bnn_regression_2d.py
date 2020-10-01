import argparse

import numpy as np
import torch

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from prob_mbrl import models, utils


def main():
    # model parameters
    parser = argparse.ArgumentParser("BNN regression example")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--net_shape',
                        type=lambda s: [int(d) for d in s.split(',')],
                        default=[200, 200, 200, 200])
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_components', type=int, default=5)
    parser.add_argument('--N_batch', type=int, default=100)
    parser.add_argument('--train_iters', type=int, default=15000)
    parser.add_argument('--noise_level', type=float, default=1e-3)
    parser.add_argument('--resample', action='store_true')
    parser.add_argument('--use_cuda', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    torch.set_flush_denormal(True)

    if args.use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # create toy dataset
    def f2d(X, multimodal=False):
        x, y = X[:, 0:1], X[:, 1:2]
        z = (1 + ((1 - x)**2 + (y - x**2)**2)).log()
        return torch.cat([z.sin(), z.cos()], -1)

    X = torch.stack(
        torch.meshgrid(torch.linspace(-0.35, 0.35, 25),
                       torch.linspace(-0.35, 0.35, 25))).T.reshape(-1, 2)
    X = torch.cat([
        X, X + torch.tensor([0.75, 0.75]), X + torch.tensor([0.75, -0.75]),
        X + torch.tensor([-0.75, -0.75]), X + torch.tensor([-0.75, 0.75])
    ], 0)
    Y = f2d(X)
    Y += torch.randn_like(Y) * args.noise_level
    xx = torch.stack(
        torch.meshgrid(torch.linspace(-1.5, 1.5, 50),
                       torch.linspace(-1.5, 1.5, 50))).T.reshape(-1, 2)
    yy = f2d(xx)

    fig = plt.figure()
    ax1 = fig.add_subplot(211, projection='3d')
    xx_, yy_ = xx.cpu(), yy.cpu()
    X_, Y_ = X.cpu(), Y.cpu()
    ax1.scatter(xx_[:, 0], xx_[:, 1], yy_[:, 0], s=1, alpha=0.25)
    ax1.scatter(X_[:, 0], X_[:, 1], Y_[:, 0])
    ax2 = fig.add_subplot(212, projection='3d')
    ax2.scatter(xx_[:, 0], xx_[:, 1], yy_[:, 1], s=1, alpha=0.25)
    ax2.scatter(X_[:, 0], X_[:, 1], Y_[:, 1])

    # plt.plot(xx.cpu(), -yy.cpu(), linestyle='--')
    print(('Dataset size:', X.shape[0], 'samples'))

    # # single gaussian model
    input_dims = 2
    output_dims = 2
    hids = args.net_shape

    model = models.density_network_mlp(
        input_dims, output_dims, models.GaussianDN, hids,
        [models.CDropout(args.drop_rate * torch.ones(hid))
         for hid in hids], models.activations.hhSinLU)
    model.set_scaling(X, Y)
    print(model)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    utils.train_model(model,
                      X,
                      Y,
                      n_iters=args.train_iters,
                      opt=opt,
                      resample=args.resample,
                      batch_size=args.N_batch)
    print(model)

    # # mixture of gaussians model
    nc = args.n_components
    mmodel = models.mixture_density_network_mlp(
        input_dims, output_dims, nc, models.GaussianMDN, hids,
        [models.CDropout(args.drop_rate * torch.ones(hid))
         for hid in hids], models.activations.hhSinLU)
    mmodel.set_scaling(X, Y)
    print(mmodel)

    opt = torch.optim.Adam(mmodel.parameters(), lr=args.lr)
    utils.train_model(mmodel,
                      X,
                      Y,
                      n_iters=args.train_iters,
                      opt=opt,
                      resample=args.resample,
                      batch_size=args.N_batch)
    print(mmodel)

    # plot results for single gaussian model
    xx = torch.stack(
        torch.meshgrid(torch.linspace(-2.75, 2.75, 50),
                       torch.linspace(-2.75, 2.75, 50))).T.reshape(-1, 2)
    yy = f2d(xx)
    xx_ = xx[:, None].repeat(1, 10, 1)
    with torch.no_grad():
        model.resample()
        py, py_params = model(xx_, temperature=1.0, resample=False)
        noiseless_py, noiseless_py_params = model(xx_,
                                                  temperature=1.0e-9,
                                                  resample=False)

    xx_, yy_ = xx.cpu(), yy.cpu()
    colors = np.array(list(plt.cm.rainbow_r(np.linspace(0, 1, 255))))
    fig = plt.figure(figsize=(16, 9))
    fig.canvas.set_window_title('Single Gaussian output density')
    samples = py.sample()
    noiseless_samples = noiseless_py.sample()

    for i in range(yy.shape[-1]):
        ax1 = fig.add_subplot(int(f'{yy.shape[-1]}1{i+1}'), projection='3d')
        ax1.scatter(xx_[:, 0], xx_[:, 1], yy_[:, i], s=2, alpha=0.25)
        ax1.scatter(X_[:, 0], X_[:, 1], Y_[:, i])
        ax1.scatter(xx_[:, 0:1].repeat(1, samples.shape[1]).view(-1),
                    xx_[:, 1:2].repeat(1, samples.shape[1]).view(-1),
                    noiseless_samples[..., i].view(-1),
                    s=2,
                    c=colors[0:1],
                    alpha=0.05)
        ax1.scatter(xx_[:, 0:1].repeat(1, samples.shape[1]),
                    xx_[:, 1:2].repeat(1, samples.shape[1]),
                    samples[..., i].view(-1),
                    s=2,
                    c=colors[0:1],
                    alpha=0.05)
        ax1.set_zlim3d(yy[:, i].min(), yy[:, i].max())

    # plot results for gaussian mixture model
    xx = torch.stack(
        torch.meshgrid(torch.linspace(-2.75, 2.75, 50),
                       torch.linspace(-2.75, 2.75, 50))).T.reshape(-1, 2)
    yy = f2d(xx)
    xx_ = xx[:, None].repeat(1, 10, 1)
    with torch.no_grad():
        mmodel.resample()
        py, py_params = mmodel(xx_, temperature=1.0, resample=False)
        noiseless_py, noiseless_py_params = mmodel(xx_,
                                                   temperature=1.0e-9,
                                                   resample=False)

    xx_, yy_ = xx.cpu(), yy.cpu()
    colors = np.array(list(plt.cm.rainbow_r(np.linspace(0, 1, 255))))
    samples = py.sample()
    noiseless_samples = py.sample()
    fig = plt.figure(figsize=(16, 9))
    fig.canvas.set_window_title('Mixture of Gaussians output density')
    samples = py.sample()
    noiseless_samples = noiseless_py.sample()
    logit_pi = py_params['logit_pi'].squeeze(-1).cpu()
    noiseless_logit_pi = noiseless_py_params['logit_pi'].squeeze(-1).cpu()

    for i in range(yy.shape[-1]):
        ax1 = fig.add_subplot(int(f'{yy.shape[-1]}1{i+1}'), projection='3d')
        mu = samples[:, :, i].mean(1).cpu()
        std = samples[:, :, i].std(1).cpu()
        ax1.scatter(xx_[:, 0], xx_[:, 1], yy_[:, i], s=2, alpha=0.25)
        ax1.scatter(X_[:, 0], X_[:, 1], Y_[:, i])
        ax1.scatter(xx_[:, 0:1].repeat(1, samples.shape[1]).view(-1),
                    xx_[:, 1:2].repeat(1, samples.shape[1]).view(-1),
                    noiseless_samples[..., i].view(-1),
                    s=20,
                    c=noiseless_logit_pi.argmax(-1).view(-1),
                    alpha=0.01)
        ax1.scatter(xx_[:, 0:1].repeat(1, samples.shape[1]).view(-1),
                    xx_[:, 1:2].repeat(1, samples.shape[1]).view(-1),
                    samples[..., i].view(-1),
                    s=20,
                    c=logit_pi.argmax(-1).view(-1),
                    alpha=0.1)
        ax1.set_zlim3d(yy[:, i].min(), yy[:, i].max())

    plt.show()


if __name__ == '__main__':
    main()
