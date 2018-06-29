import tqdm
import sys
import torch
from losses import gaussian_log_likelihood
from utils import iterate_minibatches


def custom_pbar(iterable, total):
    for i, item in enumerate(iterable):
        sys.stdout.write('%d/%d \r' % (i, total))
        if i < total:
            yield item
        else:
            sys.stdout.write('\n')
            break


def train_regressor(model, iters=2000, batchsize=100,
                    resample=False, optimizer=None,
                    log_likelihood=gaussian_log_likelihood):
    X = (model.X - model.mx)/model.Sx
    Y = (model.Y - model.my)/model.Sy
    N = X.shape[0]
    M = batchsize

    if optimizer is None:
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params, 1e-3)

    pbar = tqdm.tqdm(enumerate(iterate_minibatches(X, Y, M)), total=iters)

    for i, batch in pbar:
        x, y = batch
        model.zero_grad()
        outs = model(x, normalize=False, resample=resample)
        Enlml = -log_likelihood(y, *outs).sum()/M
        loss = Enlml + model.regularization_loss()/N
        loss.backward()
        optimizer.step()
        pbar.set_description('log-likelihood of data: %f' % (-Enlml))
        if i == iters:
            pbar.close()
            break
