import gc
import numpy as np
import torch
from functools import partial
from matplotlib import pyplot as plt
from scipy.special import logsumexp
from prob_mbrl import losses, models, utils
torch.set_flush_denormal(True)
torch.set_num_threads(2)


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
    n_layers = 4
    layer_width = 200
    drop_rate = 0.5
    odims = 1
    n_components = 5
    N_batch = 100
    use_cuda = False

    # single gaussian output model
    mlp = models.mlp(1,
                     2 * odims, [layer_width] * n_layers,
                     dropout_layers=[
                         models.CDropout(drop_rate *
                                         np.random.rand(layer_width))
                         for hid in range(n_layers)
                     ])
    model = models.Regressor(mlp,
                             output_density=models.DiagGaussianDensity(odims))

    # mixture density network
    mlp2 = models.mlp(
        1,
        2 * n_components * odims + n_components + 1, [layer_width] * n_layers,
        dropout_layers=[
            models.CDropout(drop_rate * np.random.rand(layer_width))
            for i in range(n_layers)
        ])
    mmodel = models.Regressor(mlp2,
                              output_density=models.GaussianMixtureDensity(
                                  odims, n_components))

    # optimizer for single gaussian model
    opt1 = torch.optim.Adam(model.parameters(), 1e-3)

    # optimizer for mixture density network
    opt2 = torch.optim.Adam(mmodel.parameters(), 1e-3)

    # create training dataset
    train_x = np.concatenate([
        np.arange(-0.6, -0.25, 0.0005),
        np.arange(0.1, 0.25, 0.0005),
        np.arange(0.65, 1.0, 0.0005)
    ])
    train_y = f(train_x)
    train_y += 0.01 * np.random.randn(*train_y.shape)
    X = torch.from_numpy(train_x[:, None]).float()
    Y = torch.from_numpy(train_y[:, None]).float()

    model.set_dataset(X, Y)
    mmodel.set_dataset(X, Y)

    model = model.float()
    mmodel = mmodel.float()

    if use_cuda and torch.cuda.is_available():
        X = X.cuda()
        Y = Y.cuda()
        model = model.cuda()
        mmodel = mmodel.cuda()

    print(('Dataset size:', train_x.shape[0], 'samples'))

    utils.train_regressor(model,
                          iters=4000,
                          batchsize=N_batch,
                          resample=True,
                          optimizer=opt1,
                          log_likelihood=model.output_density.log_prob)
    utils.train_regressor(mmodel,
                          iters=4000,
                          batchsize=N_batch,
                          resample=True,
                          optimizer=opt2,
                          log_likelihood=mmodel.output_density.log_prob)

    # evaluate single gaussian model
    test_x = np.arange(-1.0, 1.5, 0.005)
    ret = []
    model.resample()
    for i, x in enumerate(test_x):
        x = torch.tensor(x[None]).float().to(model.X.device)
        outs = model(x.expand((N_batch, 1)), resample=False)
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

    # evaluate mixture density network
    test_x = np.arange(-1.0, 1.5, 0.005)
    ret = []
    logit_weights = []
    mmodel.resample()
    for i, x in enumerate(test_x):
        x = torch.tensor(x[None]).float().to(mmodel.X.device)
        outs = mmodel(x.expand((N_batch, 1)), resample=False)
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