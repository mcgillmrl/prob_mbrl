#!/usr/bin/env python
import argparse
import gym
import joblib
import math
import numpy as np
import pandas as pd
import prob_mbrl
import torch
import nflows
from nflows import distributions, transforms, flows
from nflows.nn import nets
from tqdm.auto import tqdm


# import streamlit as st
torch.set_flush_denormal(True)


def rollout(env, pol, seed=None, worker_id=0, render=False, stop_on_done=False, max_steps=1000):
    data = []

    # initalize the random seed differently for each worker
    seed = env.seed(seed)[0] + worker_id
    seed = env.seed(seed)[0]
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    obs = env.reset()

    # sample a couple different actions for each worker, so we end up with different action sequences
    with torch.no_grad():
        for i in range(worker_id % 10):
            act = pol(obs)

    for t in range(max_steps):
        if isinstance(pol, torch.nn.Module):
            p = list(pol.parameters())[0]
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=p.dtype, device=p.device)
            else:
                obs.to(dtype=p.dtype, device=p.device)
            if obs.ndim == 1:
                obs = obs.unsqueeze(0)
        # evaluate policy
        act = pol(obs)
        if isinstance(act, torch.distributions.Distribution):
            act = act.rsample()
        elif isinstance(act, nflows.flows.Flow):
            act = act.sample()
        if isinstance(env, torch.nn.Module):
            p = list(env.parameters())[0]
            if not isinstance(obs, torch.Tensor):
                act = torch.tensor(act, dtype=p.dtype, device=p.device)
            else:
                act.to(dtype=p.dtype, device=p.device)
        elif isinstance(act, torch.Tensor):
            # if we're passing actions to a gym env but the action is a tensor
            act = act.detach().numpy().squeeze(0)
            obs = obs.numpy().squeeze(0)

        # step environment
        next_obs, rew, done, info = env.step(act)
        data.append(dict(obs=obs, act=act, next_obs=next_obs,
                    rew=rew, done=done, info=info))
        if render:
            env.render()
        if done and stop_on_done:
            break
    return pd.DataFrame(data)


def parallel_rollout(env, pol, seed=None, n_trajs=1, n_jobs=8):
    if n_trajs == 1:
        data = [rollout(env, pol, seed)]
    else:
        n_jobs = min(n_trajs, n_jobs)
        if seed is None:
            seed = 0
        with prob_mbrl.utils.tqdm_joblib(tqdm(desc=f"rollout {env.__repr__()}", total=n_trajs)):
            data = joblib.Parallel(n_jobs=n_jobs)(
                [joblib.delayed(rollout)(env, rnd, seed=seed, worker_id=i)
                 for i in range(n_trajs)])
    for i, d in enumerate(data):
        d['trajectory_id'] = i
    return pd.concat(data)


class ExperienceDataset(torch.utils.data.Dataset):
    def __init__(self, data, planning_horizon=50, min_traj_len=None):
        self.data = data
        self.planning_horizon = planning_horizon
        self.min_traj_length = planning_horizon if min_traj_len is None else min_traj_len
        self.init_indices()

    def init_indices(self):
        self.traj_groups = self.data.groupby('trajectory_id')
        self.traj_lens = self.traj_groups.done.count()
        self.end_idxs = (self.traj_lens - (self.min_traj_length) + 1
                         ).where(lambda x: x > 0, 1).cumsum()
        self.start_idxs = (self.end_idxs.shift(1)).fillna(0).astype(int)

    def append(self, new_data):
        new_data.trajectory_id += self.data.trajectory_id.max() + 1
        self.data = pd.concat([self.data, new_data])
        self.init_indices()

    def __len__(self):
        return self.end_idxs.iloc[-1]

    def __getitem__(self, idx):
        group_idx = self.end_idxs[self.end_idxs > idx].index[0]
        start_idx = idx - self.start_idxs[group_idx]
        traj_segment = self.traj_groups.get_group(
            group_idx).iloc[start_idx:start_idx + self.planning_horizon]
        return traj_segment


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler',
                           _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # this is assuming that x is of shape [sequence length, batch size, embedding_size]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def bool_mask_to_float(mask):
    float_mask = mask.float()
    mask = torch.where(mask == 1,
                       torch.empty_like(float_mask).fill_(float('-inf')),
                       torch.zeros_like(float_mask))
    return mask


def causal_mask(src_horizon, tgt_horizon, device=None):
    mask = torch.triu(torch.ones(tgt_horizon, src_horizon, device=device), 1)
    return bool_mask_to_float(mask)


def causal_mask_from_idxs(src_idxs, tgt_idxs):
    mask = src_idxs.unsqueeze(1) < tgt_idxs.unsqueeze(0)
    return bool_mask_to_float(mask)


def seqlen_padding_mask(horizon, seqlens, offset=0, device=None):
    mask = (torch.arange(horizon, device=device)[:, None] >=
            seqlens[None, :] + offset).T
    return mask


class DynamicsModel(torch.nn.Module):
    def __init__(self, input_dims, output_distribution,
                 temporal_model=None, embedding_size=128):
        super().__init__()

        # input projections
        self.input_projs = torch.nn.ModuleList([torch.nn.Linear(
            dim, embedding_size) for dim in input_dims])

        # positional encoding
        self.pos_encoder = PositionalEncoding(embedding_size)

        # temporal model
        if temporal_model is None:
            temporal_model = transformer_model(embedding_size, 4, 4)
        self.temporal_model = temporal_model

        # output model
        self.output_distribution = output_distribution

    def forward(self, inputs, attn_masks=[], pad_masks=[], batch_first=False):
        if batch_first:
            inputs = [x.permute(1, 0, 2) for x in inputs]

        # compute the embeddings for each input type (state, action, etc)
        input_embs = [self.input_projs[i](x) for (i, x) in enumerate(inputs)]

        # apply positional embeddings to the inputs (e.g. states and actions from the same timestep
        # will be assigned the same positional encoding)
        input_embs = [self.pos_encoder(embs) for embs in input_embs]
        # concat input embeddings to for sequence of shape [sum(len(x) for x in input_embs), bsize, embedding_size]
        # e.g. if input embeddings are state and actions, then the input sequence will be
        # s_1, s_2, ..., s_horizon, a_1, a_2, ..., a_horizon
        input_embs = torch.cat(input_embs, 0)
        horizon, batch_size, embedding_size = input_embs.shape
        # combine attn masks by concatenating on source dimension and repeating over target dimension
        attn_mask = torch.cat(
            attn_masks, -1).repeat_interleave(2, -2) if len(attn_masks) == 0 else None
        # combine padding masks by concatenating on source dimension
        pad_mask = torch.cat(pad_masks, -1) if len(pad_masks) == 0 else None
        # process sequences with transformer
        output_embs = self.temporal_model(input_embs, attn_mask, pad_mask)
        # we combine the output embeddings for each input type to produce one embedding per timestep
        output_embs = output_embs.view(
            2, -1, batch_size, embedding_size).mean(0)
        # return output distribution
        prob_output = self.output_distribution(output_embs)
        return prob_output

    def step(self, act):
        pass


class NextStateRewardDoneOutputDistribution(torch.nn.Module):
    def __init__(self, embedding_size, D, hids=[], input_dropout=0.1):
        super().__init__()
        self.ps = prob_mbrl.models.density_network_mlp(
            embedding_size, D, hids=[], input_dropout=0.1)
        self.pr = prob_mbrl.models.density_network_mlp(
            embedding_size + self.ps.n_params(D), 1, hids=[], input_dropout=0.1)
        self.pdone = prob_mbrl.models.density_network_mlp(
            embedding_size + self.ps.n_params(D+1), 2, density_model=prob_mbrl.models.SoftmaxDN,
            hids=[], input_dropout=0.1)
        self.pdone.one_hot = False

    def regularization_loss(self):
        return self.ps.regularization_loss() + self.pr.regularization_loss() + self.pdone.regularization_loss()

    def forward(self, x, *kwargs):
        ps, ps_params = self.ps(x, return_params=True, *kwargs)
        x = torch.cat([x, ps_params['raw_params_vector']], -1)
        pr, pr_params = self.pr(x, return_params=True, *kwargs)
        x = torch.cat([x, pr_params['raw_params_vector']], -1)
        pdone = self.pdone(x, *kwargs)
        return ps, pr, pdone


class Policy(torch.nn.Module):
    def __init__(self, base_policy, limits=None):
        super().__init__()
        # limits must be a tensor of shape [2, action_dims] with the first tensor
        # corresponding to the low action limit and the second one to the high limit
        self.register_buffer('limits', limits)
        self.base_policy = base_policy

        # initialize policy  parameters to very small numbers
        for p in self.parameters():
            p.data *= 1e-3

    def regularization_loss(self):
        return self.base_policy.regularization_loss()

    def forward(self, s, **kwargs):
        # base policy must return a torch.distributions.Distribution object
        p_act = self.base_policy(s, *kwargs)
        transforms = [torch.distributions.SigmoidTransform(), torch.distributions.AffineTransform(
            loc=self.limits[0], scale=self.limits[1]-self.limits[0])]
        p_act = torch.distributions.TransformedDistribution(p_act, transforms)
        return p_act


def transformer_model(embedding_size, n_heads, n_layers, fc_dim=None):
    if fc_dim is None:
        fc_dim = 2 * embedding_size
    encoder_layer = torch.nn.TransformerEncoderLayer(
        embedding_size, n_heads, fc_dim)
    transformer_model = torch.nn.TransformerEncoder(encoder_layer, n_layers)
    return transformer_model


def nf_model(num_layers=2, hids=100, dims=2, context_dims=2, batch_norm=False, activation=torch.nn.functional.relu):
    # initialize normalizing flow model
    if context_dims is None:
        base_dist = nflows.distributions.StandardNormal(shape=[dims])
    else:
        base_dist = nflows.distributions.ConditionalDiagonalNormal(
            shape=[dims], context_encoder=torch.nn.Linear(context_dims, 2*dims))

    transforms = []

    for _ in range(num_layers):
        transforms.append(nflows.transforms.ReversePermutation(features=dims))
        # affine flows
        transforms.append(nflows.transforms.MaskedAffineAutoregressiveTransform(
            features=dims,
            hidden_features=hids,
            context_features=context_dims,
            use_batch_norm=batch_norm,
            activation=activation
        ))

    transform = nflows.transforms.CompositeTransform(transforms)

    class FlowWithDefaultContext(nflows.flows.Flow):
        def __init__(self, transform, base_dist, context_dims):
            super().__init__(transform, base_dist)
            self.context_dims = context_dims

        def get_default_context(self, context=None):
            if context is None:
                device = self._distribution._log_z.device
                context = torch.zeros([1, self.context_dims], device=device)
            return context

        def sample(self, num_samples, context=None):
            return super().sample(num_samples, self.get_default_context(context))

        def log_prob(self, inputs, context=None):
            return super().log_prob(inputs, self.get_default_context(context))

        def sample_and_log_prob(self, inputs, context=None):
            return super().log_prob(inputs, self.get_default_context(context))

    flow = FlowWithDefaultContext(transform, base_dist, context_dims)
    return flow


def polar_coords_noise(x):
    directions = torch.randn_like(x)
    norms = ((directions**2).sum(-1)[:, None])**0.5
    directions = directions/norms
    magnitudes = torch.randn_like(x)
    samples = directions*magnitudes
    return samples


class DynCollatePadRandomInputDrop(torch.nn.Module):
    def __init__(self, p_full_obs=0.5, p_full_act=1.0,  p_drop_obs=0.5, p_drop_act=0.0):
        super().__init__()
        self.p_full_obs = p_full_obs
        self.p_full_act = p_full_act
        self.p_drop_obs = p_drop_obs
        self.p_drop_act = p_drop_act

    def drop_mask(self, seqs, p_full, p_drop):
        return [torch.zeros(s.shape[0]).bool()
                if full < p_full
                else (torch.arange(s.shape[0]) > 0) & (torch.rand(s.shape[0]) < p_drop)
                for (s, full) in zip(seqs, torch.rand(len(seqs)))]

    def forward(self, batch):
        obs = [torch.tensor(np.stack(b.obs)).float() for b in batch]
        act = [torch.tensor(np.stack(b.act)).float() for b in batch]
        next_obs = [torch.tensor(np.stack(b.next_obs)).float() for b in batch]
        rew = [torch.tensor(np.stack(b.rew)).float() for b in batch]
        done = [torch.tensor(np.stack(b.done)).float() for b in batch]

        # sample which indices we want to drop from input (by masking)
        obs_drop_mask = self.drop_mask(obs, self.p_full_obs, self.p_drop_obs)
        act_drop_mask = self.drop_mask(obs, self.p_full_act, self.p_drop_act)

        # pad sequences of different lengths
        seqlens = torch.tensor([len(o) for o in obs]).long()
        obs = torch.nn.utils.rnn.pad_sequence(obs)
        act = torch.nn.utils.rnn.pad_sequence(act)
        next_obs = torch.nn.utils.rnn.pad_sequence(next_obs)
        rew = torch.nn.utils.rnn.pad_sequence(rew)
        done = torch.nn.utils.rnn.pad_sequence(done)
        obs_drop_mask = torch.nn.utils.rnn.pad_sequence(
            obs_drop_mask).T
        act_drop_mask = torch.nn.utils.rnn.pad_sequence(
            act_drop_mask).T

        horizon = obs.shape[0]
        attn_mask = causal_mask(horizon, horizon, device=obs.device)
        pad_mask = seqlen_padding_mask(
            horizon, seqlens, device=obs.device)

        return dict(
            obs=obs, act=act, next_obs=next_obs, rew=rew, done=done, seqlens=seqlens,
            obs_attn_mask=attn_mask, act_attn_mask=attn_mask,
            obs_pad_mask=pad_mask | obs_drop_mask, act_pad_mask=pad_mask | act_drop_mask)


def collate_initial_state(batch):
    obs0 = torch.stack([torch.tensor(b.obs.iloc[0]).float() for b in batch])
    return dict(obs0=obs0)


def minibatch_supervised_train(model, opt, dataset, loss_cb, n_iters=1000, batch_size=100, num_workers=0, use_cuda=True, collate_fn=None):
    dataloader = FastDataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
        drop_last=batch_size < len(dataset), collate_fn=collate_fn)
    pbar = tqdm(range(n_iters))
    data_iter = iter(dataloader)
    model.train()
    if use_cuda:
        model.cuda()
    for i in pbar:
        try:
            batch = next(data_iter)
            opt.zero_grad()

            # evaluate model
            loss = loss_cb(batch, model, use_cuda)

            # update params
            loss.backward()
            opt.step()
            if i % 10 == 0:
                pbar.set_description(
                    f"E_nll {loss.detach().cpu().numpy():0.4f}")
        except StopIteration:
            data_iter = iter(dataloader)
    model.eval()
    model.cpu()


def dyn_loss_cb(batch, model, use_cuda=False):
    obs, act, next_obs = batch['obs'], batch['act'], batch['next_obs']
    rew, done = batch['rew'], batch['done']
    obs_attn_mask, obs_pad_mask = batch['obs_attn_mask'], batch['obs_pad_mask']
    act_attn_mask, act_pad_mask = batch['act_attn_mask'], batch['act_pad_mask']

    if use_cuda:
        obs, act, next_obs = obs.cuda(), act.cuda(), next_obs.cuda()
        rew, done = rew.cuda(), done.cuda()
        obs_attn_mask, obs_pad_mask = obs_attn_mask.cuda(), obs_pad_mask.cuda()
        act_attn_mask, act_pad_mask = act_attn_mask.cuda(), act_pad_mask.cuda()

    p_next_obs, p_rew, p_done = model(
        (obs, act), (obs_attn_mask, obs_attn_mask), (obs_pad_mask, obs_pad_mask))

    log_p_next_obs = p_next_obs.log_prob(next_obs)
    log_p_rew = p_rew.log_prob(rew.unsqueeze(-1))
    log_p_done = p_done.log_prob(done)

    log_p_traj = log_p_next_obs.sum(0) + log_p_rew.sum(0) + log_p_done.sum(0)
    return -log_p_traj.mean() + 1e-3*dyn.output_distribution.regularization_loss()


def p0_loss_cb(batch, model, use_cuda=False):
    obs0 = batch['obs0']
    if use_cuda:
        obs0 = obs0.cuda()
    # sample noise regularizer magnitude
    h = 2*torch.rand_like(obs0[..., 0:1])
    # perturb samples with noise proportional to the std of the data
    obs0_noisy = obs0 + h*obs0.std(0, keepdim=True)
    # compute log prob of corrupted samples
    log_p0 = model.log_prob(obs0_noisy, context=h)
    return -log_p0.mean()


def train_policy_on_rollouts(pol, dyn, p0, opt, pol_reg, n_iters=1000, batch_size=100, use_cuda=True):
    if use_cuda == True:
        p0.cuda()
        pol.cuda()
        dyn.cuda()
    p0.eval()
    pol.train()
    dyn.eval()
    pbar = tqdm(range(max(1, n_iters)))
    pbar_iter = iter(pbar)
    i = next(pbar_iter)
    while i < n_iters:
        # initialize state distribution
        # sample initial states
        s0 = p0.sample(batch_size)
        s = s0.detach()
        while s.shape[0] < planning_horizon:
            # compute attention masks
            amask = causal_mask(s.shape[0], s.shape[0], device=s.device)
            pmask = torch.empty(
                pol_batch_size, s.shape[0], device=s.device, dtype=torch.bool).fill_(False)
            dyn.zero_grad()
            p0.zero_grad()
            polopt.zero_grad()

            # evaluate policy
            p_a = pol(s)
            a = p_a.rsample()

            # compute next state
            p_next_s, p_rew, p_done = dyn(
                [s, a], [amask, amask], [pmask, pmask])
            next_s = p_next_s.rsample()

            # compute loss and update policy
            E_r = p_rew.rsample().sum(0).mean()
            loss = -E_r + pol_reg*pol.regularization_loss()
            loss.backward(inputs=list(pol.parameters()))
            opt.step()
            s = torch.cat([s0, next_s]).detach()
            if i % 10 == 0:
                pbar.set_description(
                    f"Cumm. Rewards: {E_r.detach().cpu().numpy():0.4f}")
            try:
                i = next(pbar_iter)
            except StopIteration:
                i += 1
                break

    p0.cpu()
    pol.cpu()
    dyn.cpu()
    p0.eval()
    pol.eval()
    dyn.eval()


if __name__ == '__main__':
    env_name = 'Hopper-v2'
    n_initial_trajs = 10
    trajs_per_iter = 4
    dyn_batch_size = 100
    dyn_iters = 1000
    dyn_reg = 1e-3
    p0_batch_size = 100
    p0_iters = 1000
    pol_batch_size = 100
    pol_iters = 1000
    pol_reg = 1e-3
    num_workers = 0
    embedding_size = 128
    planning_horizon = 50
    use_cuda = True

    env = gym.make(env_name)
    D = env.observation_space.high.flatten().shape[0]
    U = env.action_space.high.flatten().shape[0]
    limits = torch.tensor([env.action_space.low, env.action_space.high])
    def rnd(x): return env.action_space.sample()

    # dyn model

    dyn = DynamicsModel(
        (D, U),
        NextStateRewardDoneOutputDistribution(
            embedding_size, D, hids=[], input_dropout=0.1),
        transformer_model(embedding_size, 4, 4),
        embedding_size)
    dynopt = torch.optim.Adam(dyn.parameters(), lr=1e-3)

    # initial state model
    p0 = nf_model(dims=D, context_dims=1)
    p0opt = torch.optim.Adam(p0.parameters(), lr=1e-3)

    # policy
    limits = torch.tensor([env.action_space.low, env.action_space.high])
    squash_transforms = [torch.distributions.SigmoidTransform(),
                         torch.distributions.AffineTransform(loc=limits[0], scale=limits[1]-limits[0])]
    pol = Policy(prob_mbrl.models.density_network_mlp(D, U),  limits)
    polopt = torch.optim.Adam(pol.parameters(), lr=1e-3)

    # collect dataset of 10 trajectories
    data = parallel_rollout(env, rnd, n_trajs=n_initial_trajs)
    dataset = ExperienceDataset(data, planning_horizon, min_traj_len=2)

    for i in range(10):
        # show what the current policy can do
        rollout(env, pol, seed=env.seed(None)[0], render=True)

        # train initial state model
        minibatch_supervised_train(p0, p0opt, dataset, p0_loss_cb, n_iters=p0_iters,
                                   batch_size=p0_batch_size, num_workers=num_workers,
                                   collate_fn=collate_initial_state)

        # train dynamics model
        minibatch_supervised_train(dyn, dynopt, dataset, dyn_loss_cb, n_iters=dyn_iters,
                                   batch_size=dyn_batch_size, num_workers=num_workers,
                                   collate_fn=DynCollatePadRandomInputDrop())

        # train policy on dynamics model rollouts
        train_policy_on_rollouts(pol, dyn, p0, polopt, pol_reg,
                                 pol_iters, pol_batch_size, use_cuda)

        # collect new data
        data = parallel_rollout(env, pol, n_trajs=trajs_per_iter)
        dataset.append(data)
