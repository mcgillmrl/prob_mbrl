import argparse
import gc
import numpy as np
import torch

from matplotlib import pyplot as plt
from scipy.special import logsumexp
from prob_mbrl import models, utils


def main():
    # model parameters
    parser = argparse.ArgumentParser("BNN regression example")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--net_shape',
                        type=lambda s: [int(d) for d in s.split(',')],
                        default=[200, 200])
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_components', type=int, default=5)
    parser.add_argument('--N_batch', type=int, default=100)
    parser.add_argument('--train_iters', type=int, default=10000)
    parser.add_argument('--noise_level', type=float, default=0)
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
    def f(x, multimodal=False):
        c = 100
        if multimodal:
            c *= 2 * torch.randint_like(x, 2) - 1
        return c * sum([
            torch.sin(-2 * np.pi * (2 * k - 1) * x) / (2 * k - 1)
            for k in range(1, 3)
        ])

    # create training dataset
    train_x = torch.cat([
        torch.arange(-0.6, -0.25, 0.01),
        torch.arange(0.1, 0.45, 0.005),
        torch.arange(0.7, 1.25, 0.01)
    ])
    train_y = f(train_x, False)
    train_y += 0.01 * torch.randn(*train_y.shape)
    X = train_x[:, None]
    Y = train_y[:, None]
    Y = Y + torch.randn_like(Y) * args.noise_level
    plt.scatter(X.cpu(), Y.cpu())
    xx = torch.linspace(-.1 + X.min(), .1 + X.max())
    yy = f(xx)
    plt.plot(xx.cpu(), yy.cpu(), linestyle='--')
    # plt.plot(xx.cpu(), -yy.cpu(), linestyle='--')
    print(('Dataset size:', train_x.shape[0], 'samples'))

    # single gaussian model
    input_dims = 1
    output_dims = 1
    hids = args.net_shape
    net = models.mlp(input_dims,
                     models.GaussianDN.n_params(output_dims),
                     hids,
                     dropout_layers=[
                         models.CDropout(args.drop_rate * torch.ones(hid))
                         for hid in hids
                     ],
                     nonlin=models.activations.hhSinLU)
    model = models.GaussianDN(net)
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

    # mixture of gaussians model
    nc = args.n_components
    net = models.mlp(input_dims,
                     models.GaussianMDN.n_params(output_dims, nc),
                     hids,
                     dropout_layers=[
                         models.CDropout(args.drop_rate * torch.ones(hid))
                         for hid in hids
                     ],
                     nonlin=models.activations.hhSinLU)
    mmodel = models.GaussianMDN(net, nc)
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
    xx = torch.linspace(-2.5 + X.min(), 2.5 + X.max(), 500)
    xx = xx[:, None, None].repeat(1, 100, 1)
    with torch.no_grad():
        model.resample()
        yy_pred = model(xx, temperature=1.0, resample=False)
        noiseless_yy_pred = model(xx, temperature=1.0e-9, resample=False)
    xx = xx.cpu()
    colors = np.array(list(plt.cm.rainbow_r(np.linspace(0, 1, 255))))

    fig = plt.figure(figsize=(16, 9))
    means = yy_pred[2]['mu'].squeeze(-1).cpu()
    stds = yy_pred[2]['sqrtSigma'].diagonal(0, -1, -2).squeeze(-1).cpu()
    plt.plot(xx.squeeze(-1), means, c=colors[0], alpha=0.1)
    for i in range(means.shape[1]):
        plt.fill_between(xx[:, i].squeeze(-1),
                         means[:, i] - stds[:, i],
                         means[:, i] + stds[:, i],
                         color=0.5 * colors[0],
                         alpha=0.1)
    plt.scatter(X.cpu(), Y.cpu())
    yy = f(xx[:, 0]).cpu()
    plt.plot(xx[:, 0], yy, linestyle='--')
    plt.ylim(1.5 * yy.min(), 1.5 * yy.max())

    # plot results for gaussian mixture model
    xx = torch.linspace(-2.5 + X.min(), 2.5 + X.max(), 500)
    xx = xx[:, None, None].repeat(1, 100, 1)
    with torch.no_grad():
        mmodel.resample()
        yy_pred = mmodel(xx, temperature=1.0, resample=False)
        noiseless_yy_pred = mmodel(xx, temperature=1.0e-9, resample=False)
    xx = xx.cpu()

    fig = plt.figure(figsize=(16, 9))
    ax = fig.gca()
    colors = np.array(list(plt.cm.rainbow_r(np.linspace(0, 1, 255))))
    means = yy_pred[2]['mu'].squeeze(-1).cpu()
    stds = yy_pred[2]['sqrtSigma'].diagonal(0, -1, -2).squeeze(-1).cpu()
    logit_pi = yy_pred[2]['logit_pi'].squeeze(-1).cpu()

    # plot samples from the mixture
    pi = torch.log_softmax(logit_pi, -1).exp()
    comp = pi.max(-1, keepdim=True)[1]
    ret = plt.scatter(xx.squeeze(-1),
                      yy_pred[0].squeeze(-1).cpu(),
                      c=0.5 * colors[0:1],
                      alpha=0.1,
                      s=1)
    ret = plt.scatter(xx.squeeze(-1),
                      noiseless_yy_pred[0].squeeze(-1).cpu(),
                      c=colors[0:1],
                      alpha=0.1,
                      s=1)

    plt.scatter(X.cpu(), Y.cpu())
    yy = f(xx[:, 0]).cpu()
    plt.plot(xx[:, 0], yy, linestyle='--')
    ret = plt.ylim(1.5 * yy.min(), 1.5 * yy.max())

    plt.show()


if __name__ == '__main__':
    main()
