import numpy as np
import sys
import torch

from prob_mbrl.losses import gaussian_log_likelihood
from tqdm import tqdm


def iterate_minibatches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)
    N = len(inputs)
    indices = np.arange(0, N)
    np.random.shuffle(indices)
    while True:
        for i in range(0, len(inputs), batchsize):
            idx = indices[i:i + batchsize]
            yield inputs[idx], targets[idx]


def custom_pbar(iterable, total):
    for i, item in enumerate(iterable):
        sys.stdout.write('%d/%d \r' % (i, total))
        if i < total:
            yield item
        else:
            sys.stdout.write('\n')
            break


def train_regressor(model,
                    iters=2000,
                    batchsize=100,
                    resample=True,
                    optimizer=None,
                    log_likelihood=gaussian_log_likelihood,
                    pbar_class=tqdm):
    X = (model.X - model.mx) * model.iSx
    Y = (model.Y - model.my) * model.iSy
    N = torch.tensor(X).shape[0]
    M = batchsize
    model.train()

    if optimizer is None:
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params, 1e-4, amsgrad=True)

    pbar = pbar_class(enumerate(iterate_minibatches(X, Y, M)), total=iters)

    for i, batch in pbar:
        x, y = batch
        model.zero_grad()
        outs = model(x, normalize=False, resample=resample)
        Enlml = -log_likelihood(y, *outs).mean()
        loss = Enlml + model.regularization_loss() / N
        loss.backward()
        optimizer.step()
        pbar.set_description('log-likelihood of data: %f' % (-Enlml))
        if i == iters:
            pbar.close()
            break
    model.eval()
