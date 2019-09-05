import collections
import numpy as np
import os
import torch

from . import angles


class ExperienceDataset(torch.nn.Module):
    ''' Class used to store data from runs with a learning agent'''

    def __init__(self, name='Experience'):
        super(ExperienceDataset, self).__init__()
        self.name = name
        self.time_stamps = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.done = []
        self.info = []
        self.policy_parameters = []
        self.curr_episode = -1
        self.state_changed = True

    def add_sample(self,
                   x_t=None,
                   u_t=None,
                   c_t=None,
                   done=None,
                   info=None,
                   t=None):
        '''
            Adds new set of observations to the current episode
        '''
        curr_episode = self.curr_episode
        if curr_episode < 0:
            self.new_episode()
        self.states[curr_episode].append(x_t)
        self.actions[curr_episode].append(u_t)
        self.rewards[curr_episode].append(c_t)
        self.done[curr_episode].append(done)
        self.info[curr_episode].append(info)
        self.time_stamps[curr_episode].append(t)
        self.state_changed = True

    def new_episode(self, policy_params=None):
        '''
            Adds new episode to the experience dataset
        '''
        self.time_stamps.append([])
        self.states.append([])
        self.actions.append([])
        self.rewards.append([])
        self.info.append([])
        if policy_params:
            self.policy_parameters.append(policy_params)
        else:
            self.policy_parameters.append([])

        self.curr_episode += 1
        self.state_changed = True

    def append_episode(self,
                       states,
                       actions,
                       rewards,
                       dones=None,
                       infos=None,
                       policy_params=None,
                       ts=None):
        if policy_params is not None:
            self.policy_parameters.append(policy_params)
        if dones is not None:
            self.done.append(dones)
        if infos is not None:
            self.info.append(infos)
        if ts is not None:
            self.time_stamps.append(ts)
        self.states.append(states)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.curr_episode += 1

    def n_samples(self):
        ''' Returns the total number of samples in this dataset '''
        return sum([len(s) for s in self.states])

    def n_episodes(self):
        ''' Returns the total number of episodes in this dataset '''
        return len(self.states)

    def reset(self):
        ''' Empties the internal data structures'''
        fmt = 'Resetting experience dataset'
        fmt += '(WARNING: data from %s will be overwritten)'
        # utils.print_with_stamp(fmt % (self.filename), self.name)
        self.time_stamps = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.info = []
        self.policy_parameters = []
        self.curr_episode = -1
        # Let's give people a last chance of recovering their data. Also, we
        # don't want to save an empty experience dataset
        self.state_changed = False

    def truncate(self, episode):
        ''' Resets the experience to start from the given episode number'''
        if episode <= self.curr_episode and episode > 0:
            fmt = 'Resetting experience dataset to episode %d'
            fmt += ' (WARNING: data from %s will be overwritten)'
            # utils.print_with_stamp(fmt % (episode, self.filename), self.name)
            self.curr_episode = episode
            self.time_stamps = self.time_stamps[episode:]
            self.states = self.states[episode:]
            self.actions = self.actions[episode:]
            self.rewards = self.rewards[episode:]
            self.done = self.done[episode:]
            self.info = self.info[episode:]
            self.policy_parameters = self.policy_parameters[episode:]

    def get_dynmodel_dataset(self,
                             deltas=True,
                             filter_episodes=None,
                             angle_dims=None,
                             x_steps=1,
                             u_steps=1,
                             output_steps=1,
                             return_costs=False,
                             stack=False):
        '''
        Returns a dataset where the inputs are state_actions and the outputs
        are next steps.
        Parameters:
        -----------
        deltas: wheter to return changes in state
                (x_t - x_{t-1}, x_{t-1} - x_{t-2}, ...)
                or future states
                (x_t, x_{t-1}, x_{t-2}, ...), in the output
        filter_episodes: list containing  episode indices to extract from
                         which to extract data.
                         if list empty or undefined ( equal to None ),
                         extracts data from all episodes
        angle_dims: indices of input state dimensions to linearize, by
                    converting to complex
                    representation \theta => (sin(\theta), cos(\theta))
        x_steps: how many steps in the past to concatenate as input
        u_steps: how many steps in the past to concatenate as input
        output_steps: how many steps in the future to concatenate as output
        return_costs: whether to append the cost feedback to the output
        stack: whether to stack or concatenate the multi step data
        Returns:
        --------
        X: if stack is False X is a numpy array of shape
           [n, x_steps*D + u_steps*U], where n is the number of data samples,
           D the input state dimensions.abs if stack is True, the shape of X
           is [n, x_steps, D + U]
        '''
        filter_episodes = filter_episodes or []
        angle_dims = angle_dims or []
        inputs, targets = [], []
        join = torch.stack if stack else torch.cat
        if stack:
            # ignore the u_steps parameter
            u_steps = x_steps
            # output steps
            output_steps = x_steps + output_steps - 1

        if not isinstance(filter_episodes, list):
            filter_episodes = [filter_episodes]
        if len(filter_episodes) < 1:
            # use all data
            filter_episodes = list(range(self.n_episodes()))
        for epi in filter_episodes:
            if len(self.states[epi]) == 0:
                continue
            # get state action pairs for current episode
            states, actions = torch.tensor(
                self.states[epi]).double(), torch.tensor(
                    self.actions[epi]).double()
            # convert input angle dimensions to complex representation
            states_ = angles.to_complex(states, angle_dims)
            # pad with initial state for the first x_steps timesteps
            states_ = torch.cat([states_[[0] * (x_steps - 1)], states_], 0)
            # get input states up to x_steps in the past.
            states_ = join([
                states_[i:i - x_steps - (output_steps - 1), :]
                for i in range(x_steps)
            ],
                           dim=1)
            # same for actions (u_steps in the past, pad with zeros for the
            # first u_steps)
            actions_ = torch.cat([
                torch.zeros((u_steps - 1, actions.shape[1])).double(), actions
            ])
            actions_ = join([
                actions_[i:i - u_steps - (output_steps - 1), :]
                for i in range(u_steps)
            ],
                            dim=1)

            # create input vector
            inp = torch.cat([states_, actions_], dim=-1)

            # get output states up to output_steps in the future
            H = states.shape[0]
            ostates = join([
                states[i:H - (output_steps - i - 1), :]
                for i in range(output_steps)
            ],
                           dim=1)

            #  create output vector
            tgt = (ostates[1:, :] -
                   ostates[:-1, :] if deltas else ostates[1:, :])

            # append rewards if requested
            if return_costs:
                rewards = torch.tensor(self.rewards[epi]).double().squeeze(-1)
                if rewards.dim() == 1:
                    rewards = rewards.unsqueeze(1)
                ocosts = join([
                    rewards[i:H - (output_steps - i - 1), :]
                    for i in range(output_steps)
                ],
                              dim=1)

                tgt = torch.cat([tgt, ocosts[:-1, :]], dim=-1)

            inputs.append(inp)
            targets.append(tgt)

        ret = torch.cat(inputs).detach(), torch.cat(targets).detach()
        return ret

    def sample_states(self, n_samples=1, timestep=0):
        # collect initial states
        if timestep is None:
            x0 = np.concatenate(self.states)
        else:
            if not isinstance(timestep, collections.Iterable):
                timestep = [timestep]

            x0 = np.concatenate([[ep[t] for t in timestep if t < len(ep)]
                                 for ep in self.states])

        # sample indices
        idx = np.random.choice(range(len(x0)), n_samples)
        return torch.tensor(x0)[idx].double()

    def save(self, filename):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError:
            pass
        state_dict = dict(states=self.states,
                          actions=self.actions,
                          rewards=self.rewards,
                          info=self.info,
                          times_stamps=self.time_stamps,
                          curr_episode=self.curr_episode,
                          policy_parameters=self.policy_parameters)
        torch.save(state_dict, filename)

    def load(self, filename):
        state_dict = torch.load(filename)
        self.__dict__.update(state_dict)


class SumTree:
    def __init__(self, max_size):
        self.max_size = max_size
        self.data = [None] * max_size
        self.sum_tree = np.zeros(2 * max_size - 1)
        self.counts = np.zeros(max_size)
        self.idx = 0
        self.max_p = 1.0
        self.max_count = 0
        self.size = 0
        self.norm_factor = 1.0

    def append(self, data, priority):
        self.data[self.idx] = data
        self.counts[self.idx] = 0
        self.update(self.idx + self.max_size - 1, priority)
        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def update(self, idx, priority):
        self.sum_tree[idx] = priority * self.norm_factor
        self._update(idx)
        self.max_p = max(self.max_p, priority)

    def renormalize(self):
        return
        # update_nf
        nf_change = 1.0 / (self.sum_tree[0])
        self.norm_factor *= nf_change
        # renormalize
        self.sum_tree *= nf_change

    def _update(self, idx):
        while idx != 0:
            parent = (idx - 1) // 2
            pleft = 2 * parent + 1
            pright = pleft + 1
            self.sum_tree[
                parent] = self.sum_tree[pleft] + self.sum_tree[pright]
            idx = parent

    def _retrieve(self, idx, priority):
        N_nodes = len(self.sum_tree)
        left = 2 * idx + 1
        while left < N_nodes:
            left_val = self.sum_tree[left]
            if priority <= left_val:
                idx = left
            else:
                idx = left + 1
                priority -= left_val
            left = 2 * idx + 1
        return idx

    def get(self, priority):
        idx = self._retrieve(0, priority)
        return [idx, self.sum_tree[idx], self.data[idx - self.max_size + 1]]

    def _retrieve_batch(self, idx, priority):
        N_nodes = len(self.sum_tree)
        left = 2 * idx + 1
        priority = priority.copy()
        in_bounds = left < N_nodes
        while any(in_bounds):
            left_val = self.sum_tree[left]
            left_cond = priority <= left_val
            right_cond = np.logical_not(left_cond)
            idx = np.where(left_cond, left, left + 1)
            priority = np.where(right_cond, priority - left_val, priority)
            left = 2 * idx + 1
            in_bounds = left < N_nodes
            left = np.where(in_bounds, left, idx)
        return idx

    def get_batch(self, priority):
        priority = np.atleast_1d(priority)
        idxs = self._retrieve_batch(np.zeros(len(priority), dtype=np.int64),
                                    priority)
        data_idxs = idxs - self.max_size + 1
        return idxs, self.sum_tree[idxs], [self.data[idx] for idx in data_idxs]

    def sample(self, batchsize, beta=1.0):
        sum_p = self.sum_tree[0]
        segment_length = sum_p / batchsize
        priorities = (np.arange(batchsize) +
                      np.random.rand(batchsize)) * segment_length
        if batchsize < 32:
            idxs, priorities, samples = zip(*[self.get(p) for p in priorities])
            idxs = np.array(idxs)
        else:
            idxs, priorities, samples = self.get_batch(priorities)
        self.counts[idxs - self.max_size + 1] += 1
        self.max_count = max(self.max_count,
                             self.counts[idxs - self.max_size + 1].max())
        probs = np.array(priorities) / sum_p
        weights = (self.size * probs)**-beta
        weights = weights / weights.max()
        return samples, idxs, weights
