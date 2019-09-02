import argparse
import gc
import numpy as np
import torch
from functools import partial
from matplotlib import pyplot as plt
from scipy.special import logsumexp
from prob_mbrl import models, utils


def gaussian_sample(mu, log_sigma):
    z2 = np.random.randn(*mu.shape)
    return mu + z2 * np.exp(log_sigma)


def mixture_sample(mu, log_sigma, logit_pi, colors=None, noise=True):
    z1 = np.random.rand(*logit_pi.shape)
    g1 = -np.log(-np.log(z1))
    k = (logit_pi - logsumexp(logit_pi, -1)[:, None] + g1).argmax(-1)
    idx = np.arange(len(mu))
    samples = mu[idx, k]
    if noise:
        z2 = np.random.randn(*mu.shape[:-1])
        samples += z2 * np.exp(log_sigma[idx, k])

    if colors is not None:
        return samples, colors[k]
    return samples


def f(x, multimodal=False):
    c = 100
    if multimodal:
        c *= np.random.choice([-1, 1], x.shape[0])
    return c * sum([
        np.sin(-2 * np.pi * (2 * k - 1) * x) / (2 * k - 1)
        for k in range(1, 3)
    ])


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
    parser.add_argument('--noise_level', type=float, default=1e-1)
    parser.add_argument('--resample', action='store_true')
    parser.add_argument('--use_cuda', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)

    idims, odims = 1, 1
    # single gaussian output model
    mlp = models.mlp(idims,
                     2 * odims,
                     args.net_shape,
                     dropout_layers=[
                         models.CDropout(args.drop_rate * np.ones(hid))
                         for hid in args.net_shape
                     ])
    model = models.Regressor(mlp,
                             output_density=models.DiagGaussianDensity(odims))

    # mixture density network
    mlp2 = models.mlp(idims,
                      2 * args.n_components * odims + args.n_components + 1,
                      args.net_shape,
                      dropout_layers=[
                          models.CDropout(args.drop_rate * np.ones(hid))
                          for hid in args.net_shape
                      ])
    mmodel = models.Regressor(mlp2,
                              output_density=models.GaussianMixtureDensity(
                                  odims, args.n_components))

    # optimizer for single gaussian model
    opt1 = torch.optim.Adam(model.parameters(), args.lr)

    # optimizer for mixture density network
    opt2 = torch.optim.Adam(mmodel.parameters(), args.lr)

    # create training dataset
    train_x = np.concatenate([
        np.linspace(-0.6, -0.25, 100),
        np.linspace(0.1, 0.25, 100),
        np.linspace(0.65, 1.0, 100)
    ])
    train_y = f(train_x)
    train_y += args.noise_level * np.random.randn(*train_y.shape)
    X = torch.from_numpy(train_x[:, None]).float()
    Y = torch.from_numpy(train_y[:, None]).float()

    model.set_dataset(X, Y)
    mmodel.set_dataset(X, Y)

    model = model.float()
    mmodel = mmodel.float()

    if args.use_cuda and torch.cuda.is_available():
        X = X.cuda()
        Y = Y.cuda()
        model = model.cuda()
        mmodel = mmodel.cuda()

    print(('Dataset size:', train_x.shape[0], 'samples'))
    # train unimodal regressor
    utils.train_regressor(model,
                          iters=args.train_iters,
                          batchsize=args.N_batch,
                          resample=args.resample,
                          optimizer=opt1,
                          log_likelihood=model.output_density.log_prob)

    # evaluate single gaussian model
    test_x = np.arange(-1.0, 1.5, 0.005)
    ret = []
    if args.resample:
        model.resample()
    for i, x in enumerate(test_x):
        x = torch.tensor(x[None]).float().to(model.X.device)
        outs = model(x.expand((2 * args.N_batch, 1)), resample=False)
        y = torch.cat(outs[:2], -1)
        ret.append(y.cpu().detach().numpy())
        torch.cuda.empty_cache()
    ret = np.stack(ret)
    ret = ret.transpose(1, 0, 2)
    torch.cuda.empty_cache()
    for i in range(3):
        gc.collect()

    plt.figure(figsize=(16, 9))
    nc = ret.shape[-2]
    colors = np.array(list(plt.cm.rainbow_r(np.linspace(0, 1, nc))))
    for i in range(len(ret)):
        m, logS = ret[i, :, 0], ret[i, :, 1]
        samples = gaussian_sample(m, logS)
        plt.scatter(test_x, m, c=colors[0:1], s=1)
        plt.scatter(test_x, samples, c=colors[0:1] * 0.5, s=1)
    plt.plot(test_x, f(test_x), linestyle='--', label='true function')
    plt.scatter(X.cpu().numpy().flatten(), Y.cpu().numpy().flatten())
    plt.xlabel('$x$', fontsize=18)
    plt.ylabel('$y$', fontsize=18)

    print(model)

    # train mixture regressor
    utils.train_regressor(mmodel,
                          iters=args.train_iters,
                          batchsize=args.N_batch,
                          resample=args.resample,
                          optimizer=opt2,
                          log_likelihood=mmodel.output_density.log_prob)

    # evaluate mixture density network
    test_x = np.arange(-1.0, 1.5, 0.005)
    ret = []
    logit_weights = []
    if args.resample:
        mmodel.resample()
    for i, x in enumerate(test_x):
        x = torch.tensor(x[None]).float().to(mmodel.X.device)
        outs = mmodel(x.expand((2 * args.N_batch, 1)), resample=False)
        y = torch.cat(outs[:2], -2)
        ret.append(y.cpu().detach().numpy())
        logit_weights.append(outs[2].cpu().detach().numpy())
        torch.cuda.empty_cache()
    ret = np.stack(ret)
    ret = ret.transpose(1, 0, 2, 3)
    logit_weights = np.stack(logit_weights)
    logit_weights = logit_weights.transpose(1, 0, 2)
    torch.cuda.empty_cache()
    for i in range(3):
        gc.collect()

    plt.figure(figsize=(16, 9))
    nc = ret.shape[-1]
    colors = np.array(list(plt.cm.rainbow_r(np.linspace(0, 1, nc))))
    total_samples = []
    for i in range(len(ret)):
        m, logS = ret[i, :, 0, :], ret[i, :, 1, :]
        samples, c = mixture_sample(m, logS, logit_weights[i], colors)
        plt.scatter(test_x, samples, c=c * 0.5, s=1)
        samples, c = mixture_sample(m,
                                    logS,
                                    logit_weights[i],
                                    colors,
                                    noise=False)
        plt.scatter(test_x, samples, c=c, s=1)
        total_samples.append(samples)
    total_samples = np.array(total_samples)
    plt.plot(test_x, f(test_x), linestyle='--', label='true function')
    plt.scatter(X.cpu().numpy().flatten(), Y.cpu().numpy().flatten())
    plt.xlabel('$x$', fontsize=18)
    plt.ylabel('$y$', fontsize=18)

    print(mmodel)

    plt.show()


if __name__ == '__main__':
    main()
