#!/usr/bin/env python
from tqdm.auto import tqdm
import gym
import joblib
import math
import numpy as np
import pandas as pd
import prob_mbrl
import torch

import streamlit as st
torch.set_flush_denormal(True)


def rollout(env, pol, seed=None):
    data = []
    seed = env.seed(seed)[0]
    torch.manual_seed(seed)
    if seed is not None:
        seed = seed % (2**32 - 1)
    np.random.seed(seed)
    obs = env.reset()
    done = False
    while not done:
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
        if isinstance(act, dict) and 'dist' in act:
            act = act['dist'].rsample()
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
        data.append(dict(obs=obs, act=act, next_obs=next_obs, rew=rew, done=done, info=info))
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
                [joblib.delayed(rollout)(env, rnd, seed=seed + i)
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
        group_idx = self.end_idxs[(self.end_idxs - idx) > 0].index[0]
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
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def causal_mask(x, seqdim=1):
    seqlen = x.shape[seqdim]
    mask = (torch.triu(torch.ones(seqlen, seqlen)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(x.device)


def seqlen_padding_mask(x, seqlens, offset=0, seqdim=1):
    mask = (torch.arange(x.shape[seqdim], device=x.device)[:, None] >=
            torch.tensor(seqlens, device=x.device)[None, :] + offset).T
    return mask.to(x.device)


class DynamicsModel(torch.nn.Module):
    def __init__(self, state_dims, action_dims, output_distribution, temporal_model=None, embedding_size=128):
        super().__init__()
        # input projections
        self.action_proj = torch.nn.Linear(action_dims, embedding_size)
        self.state_proj = torch.nn.Linear(state_dims, embedding_size)

        # positional encoding
        self.pos_encoder = PositionalEncoding(embedding_size)

        # temporal model
        if temporal_model is None:
            temporal_model = torch.nn.LSTM(embedding_size, embedding_size, num_layers=4, batch_first=True)
        self.temporal_model = temporal_model

        # output model
        self.output_distribution = output_distribution

    def forward(self, states, actions, mask=None, pad_mask=None):
        state_embs = self.state_proj(states)
        action_embs = self.action_proj(actions)
        input_embs = state_embs.sigmoid() * action_embs.sigmoid()

        if isinstance(self.temporal_model, torch.nn.TransformerEncoder):
            # for the transformer, we need to swap the timestep and bath index dimmensions
            input_embs = state_embs.permute(1, 0, 2)
            # add position encodings
            input_embs = self.pos_encoder(input_embs)
            # process sequences with transformer
            output_embs = self.temporal_model(input_embs, mask, pad_mask)
            # swap batch and temporal dimensions back to original ordering
            output_embs = output_embs.permute(1, 0, 2)
        else:
            output_embs = input_embs

        # return output distribution
        prob_output = self.output_distribution(output_embs)
        return prob_output

    def step(self, act):
        pass


def transformer_model(embedding_size, n_heads, n_layers, fc_dim=None):
    if fc_dim is None:
        fc_dim = 2 * embedding_size
    encoder_layer = torch.nn.TransformerEncoderLayer(embedding_size, n_heads, fc_dim)
    transformer_model = torch.nn.TransformerEncoder(encoder_layer, n_layers)
    return transformer_model


def collate_pad(batch):
    obs = [torch.tensor(np.stack(b.obs)).float() for b in batch]
    act = [torch.tensor(np.stack(b.act)).float() for b in batch]
    next_obs = [torch.tensor(np.stack(b.next_obs)).float() for b in batch]
    rew = [torch.tensor(np.stack(b.rew)).float() for b in batch]
    done = [torch.tensor(np.stack(b.done)).int() for b in batch]
    info = [b['info'] for b in batch]
    # pad sequences of different lengths
    seqlens = [len(o) for o in obs]
    obs = torch.nn.utils.rnn.pad_sequence(obs, batch_first=True)
    act = torch.nn.utils.rnn.pad_sequence(act, batch_first=True)
    next_obs = torch.nn.utils.rnn.pad_sequence(next_obs, batch_first=True)
    rew = torch.nn.utils.rnn.pad_sequence(rew, batch_first=True)
    done = torch.nn.utils.rnn.pad_sequence(done, batch_first=True)
    # masks for transformer model
    # compute causal mask (constrains attention to only use past to predict the future)
    mask = causal_mask(obs)
    pad_mask = seqlen_padding_mask(obs, seqlens)
    return dict(
        obs=obs, act=act, next_obs=next_obs, rew=rew, done=done,
        info=info, seqlens=seqlens, mask=mask, pad_mask=pad_mask)


def collate_pad_random_drop(batch):
    batch = collate_pad(batch)
    # create a adding mask where random


def train_dyn_model(dyn, opt, dataset, n_iters=1000, batch_size=100, num_workers=0, use_cuda=True):
    dataloader = FastDataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
        drop_last=batch_size < len(dataset), collate_fn=collate_pad)
    pbar = tqdm(range(n_iters))
    data_iter = iter(dataloader)
    dyn.train()
    if use_cuda:
        dyn.cuda()
    for i in pbar:
        try:
            batch = next(data_iter)
            opt.zero_grad()
            # evaluate model
            obs, act, next_obs = batch['obs'], batch['act'], batch['next_obs']
            if use_cuda:
                obs, act, next_obs = obs.cuda(), act.cuda(), next_obs.cuda()
            p_next_obs = dyn(obs, act)['dist']
            log_p_next_obs = p_next_obs.log_prob(next_obs)
            log_p_traj = log_p_next_obs.sum(1)
            loss = -log_p_traj.mean()
            # update params
            loss.backward()
            opt.step()
            pbar.set_description(f"E_nll {loss.detach().cpu().numpy():0.4f}")
        except StopIteration:
            data_iter = iter(dataloader)
    dyn.eval()
    dyn.cpu()


if __name__ == '__main__':
    env_name = 'Hopper-v2'
    n_initial_trajs = 1
    trajs_per_iter = 1
    dyn_batch_size = 100
    dyn_iters = 1000
    num_workers = 8
    env = gym.make(env_name)

    D = env.observation_space.high.flatten().shape[0]
    U = env.action_space.high.flatten().shape[0]
    def rnd(x): return env.action_space.sample()

    pol = prob_mbrl.models.density_network_mlp(D, U)
    embedding_size = 128
    dyn = DynamicsModel(
        D, U,
        prob_mbrl.models.density_network_mlp(embedding_size, D, hids=[], input_dropout=0.1),
        transformer_model(embedding_size, 4, 4),
        embedding_size)
    opt = torch.optim.Adam(dyn.parameters(), lr=1e-3)

    # collect dataset of 10 trajectories
    data = parallel_rollout(env, rnd, n_trajs=n_initial_trajs, seed=env.seed(None)[0])
    dataset = ExperienceDataset(data, 10, 2)

    for i in range(10):
        # train dynamics modle
        train_dyn_model(dyn, opt, dataset, n_iters=dyn_iters, batch_size=dyn_batch_size, num_workers=num_workers)

        # collect new data
        data = parallel_rollout(env, pol, n_trajs=trajs_per_iter, seed=env.seed(None)[0])
        dataset.append(data)
