import numpy as np
import sys
import torch

from tqdm import tqdm
from prob_mbrl.losses import gaussian_log_likelihood
from prob_mbrl import utils
torch.set_printoptions(linewidth=200)

priority_tree = {}
decoupled_optimizers = {}


def iterate_minibatches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)
    N = len(inputs)
    indices = np.arange(0, max(N, batchsize)) % N
    np.random.shuffle(indices)
    while True:
        for i in range(0, len(inputs), batchsize):
            idx = indices[i:i + batchsize]
            yield inputs[idx], targets[idx], idx


def iterate_priority_tree(inputs, targets, batchsize, tree, warmup_iters=100):
    if len(inputs) > tree.size:
        [tree.append(i, tree.max_p) for i in range(tree.size, len(inputs))]

    iter_ = iterate_minibatches(inputs, targets, batchsize)
    beta = 0.4

    for i in range(warmup_iters):
        x, y, idxs = next(iter_)
        tree.counts[idxs] += 1
        idxs = idxs + tree.max_size - 1
        yield x, y, idxs, torch.ones(x.shape[0], 1)

    while True:
        data_idxs, idxs, weights = tree.sample(batchsize, beta=beta)
        beta = min(1.0, beta + 1e-3)
        data_idxs = np.array(data_idxs)
        x, y = inputs[data_idxs], targets[data_idxs]
        yield x, y, idxs, weights


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
                    decoupled_reg=False,
                    prioritized_sampling=True,
                    priority_eps=1e-3,
                    priority_alpha=0.6):
    global priority_tree
    model.train()
    X = (model.X - model.mx) * model.iSx
    Y = (model.Y - model.my) * model.iSy
    N = X.shape[0]
    M = batchsize
    print('train_regressor >', 'Dataset size [%d]' % int(N))
    model.train()
    if reg_weight is None:
        reg_weight = 1 / N

    params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer is None:
        # default optimizer
        optimizer = torch.optim.Adam(params, 1e-4)

    if decoupled_reg:
        # decoupled regularization optimizer
        if model not in decoupled_optimizers:
            decoupled_optimizers[model] = torch.optim.SGD(
                optimizer.param_groups)
        reg_optimizer = decoupled_optimizers[model]

    if prioritized_sampling:
        # whether to sample similar to prioritized experience replay
        tree = priority_tree.get(model, None)
        if tree is None or N > tree.size:
            old_tree = tree
            tree = utils.SumTree(2 * N)
            if old_tree is not None:
                tree.max_p = old_tree.max_p
                tree.counts[:len(old_tree.counts)] = old_tree.counts
            priority_tree[model] = tree
        pbar = pbar_class(enumerate(iterate_priority_tree(X, Y, M, tree)),
                          total=iters)
    else:
        pbar = pbar_class(enumerate(iterate_minibatches(X, Y, M)), total=iters)

    for i, batch in pbar:
        x, y = batch[:2]
        model.zero_grad()
        outs = model(x, normalize=False, resample=resample)
        log_probs = log_likelihood(y, *outs)

        if prioritized_sampling:
            idxs, weights = batch[2:]
            weights = torch.tensor(np.stack(weights)).to(X.device, X.dtype)
            tree = priority_tree[model]
            #priorities = (2 - log_probs.clamp(-1, 2).detach().cpu().numpy() +
            #              priority_eps)**priority_alpha
            a = 2
            p0 = 1 + (a - log_probs.flatten().clamp(
                -a, a).detach().cpu().numpy()) / (2 * a)
            priorities = (p0 * tree.max_count /
                          (tree.counts[idxs - tree.max_size + 1]) +
                          priority_eps)**priority_alpha
            [tree.update(idx, p) for idx, p in zip(idxs, priorities)]
            log_probs = log_probs * weights

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
    print(model)
