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
                    reg_weight=None,
                    pbar_class=tqdm,
                    summary_writer=None,
                    summary_scope='',
                    decoupled_reg=False):
    X = (model.X - model.mx) * model.iSx
    Y = (model.Y - model.my) * model.iSy
    N = X.shape[0]
    M = batchsize
    model.train()
    if reg_weight is None:
        reg_weight = 1 / N

    params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer is None:
        optimizer = torch.optim.Adam(params, 1e-4)
    if decoupled_reg:
        # decoupled regularization optimizer
        reg_optimizer = torch.optim.SGD(optimizer.param_groups)
    pbar = pbar_class(enumerate(iterate_minibatches(X, Y, M)), total=iters)

    for i, batch in pbar:
        x, y = batch
        model.zero_grad()
        outs = model(x, normalize=False, resample=resample)
        log_probs = log_likelihood(y, *outs)
        Enlml = -log_probs.mean()
        if not decoupled_reg:
            reg = reg_weight * model.regularization_loss()
            loss = Enlml + reg
        else:
            loss = Enlml
        loss.backward()
        optimizer.step()

        if decoupled_reg:
            # decoupled regularization
            model.zero_grad()
            reg = reg_weight * model.regularization_loss()
            reg.backward()
            reg_optimizer.step()

        pbar.set_description('log-likelihood of data: %f' % (-Enlml))

        if summary_writer is not None:
            plot1_name = 'training_loss'
            plot2_name = 'E_lml'
            plot3_name = 'reg_loss'
            if summary_scope is not None and len(summary_scope) > 0:
                plot1_name = '/'.join([summary_scope, plot1_name])
                plot2_name = '/'.join([summary_scope, plot2_name])
                plot3_name = '/'.join([summary_scope, plot3_name])
            summary_writer.add_scalar(plot1_name, loss, i)
            summary_writer.add_scalar(plot2_name, -Enlml, i)
            summary_writer.add_scalar(plot3_name, reg, i)

        if i == iters:
            pbar.close()
            break
    model.eval()
