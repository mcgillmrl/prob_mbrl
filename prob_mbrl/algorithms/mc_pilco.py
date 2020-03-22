import torch
import numpy as np
import tqdm

from collections import defaultdict
from prob_mbrl import utils

policy_update_counter = defaultdict(lambda: 0)
x0_tree = utils.SumTree(2**20)
episode_counter = 0


def mc_pilco(init_states,
             dynamics,
             policy,
             steps,
             opt=None,
             exp=None,
             opt_iters=1000,
             value_func=None,
             pegasus=True,
             mm_states=False,
             mm_rewards=False,
             mm_groups=None,
             maximize=True,
             clip_grad=1.0,
             cvar_eps=0.0,
             reg_weight=0.0,
             discount=None,
             on_rollout=None,
             on_iteration=None,
             step_idx_to_sample=None,
             init_state_noise=0.0,
             resampling_period=499,
             prioritized_replay=False,
             priority_alpha=0.6,
             priority_eps=1e-8,
             init_priority_beta=1.0,
             priority_beta_increase=0.0,
             debug=False,
             rollout_kwargs={}):
    global policy_update_counter, x0_tree, episode_counter
    dynamics.eval()
    policy.train()

    if discount is None:
        discount = lambda i: 1.0 / steps  # noqa: E731
    elif not callable(discount):
        discount_factor = discount
        discount = lambda i: discount_factor**i  # noqa: E731

    msg = "Pred. Cumm. rewards: %f" if maximize else "Pred. Cumm. costs: %f"
    if opt is None:
        params = filter(lambda p: p.requires_grad, policy.parameters())
        opt = torch.optim.Adam(params)
    pbar = tqdm.tqdm(range(opt_iters), total=opt_iters)
    D = init_states.shape[-1]
    shape = init_states.shape
    z_mm = torch.randn(steps + shape[0], *shape[1:])
    z_mm = z_mm.reshape(-1, D).to(dynamics.X.device, dynamics.X.dtype)
    z_rr = torch.randn(steps + shape[0], 1)
    z_rr = z_rr.reshape(-1, 1).to(dynamics.X.device, dynamics.X.dtype)

    def resample():
        seed = torch.randint(2**32, [1])
        dynamics.resample(seed=seed)
        policy.resample(seed=seed)
        if value_func is not None:
            value_func.resample(seed=seed)
        z_mm.normal_()
        z_rr.normal_()

    # sample initial random numbers
    resample()

    x0 = init_states
    N_particles = init_states.shape[0]
    n_opt_steps = policy_update_counter[policy]

    if prioritized_replay:
        x0_idxs = None
        x0_weights = torch.ones_like(x0)
        priority_beta = init_priority_beta
        # old_counts = x0_tree.counts.copy()

    for i in pbar:
        # zero gradients
        policy.zero_grad()
        dynamics.zero_grad()
        opt.zero_grad()
        if not pegasus or n_opt_steps % resampling_period == 0:
            resample()

        # rollout policy
        H = steps
        try:
            x0_ = x0
            if mm_groups is not None and x0_.shape[0] == mm_groups:
                x0_ = utils.tile(x0_, int(N_particles / mm_groups))
            x0_ = x0_ + init_state_noise * torch.randn_like(x0_)
            trajectories = utils.rollout(x0_,
                                         dynamics,
                                         policy,
                                         H,
                                         resample_state_noise=not pegasus,
                                         resample_action_noise=not pegasus,
                                         mm_states=mm_states,
                                         mm_rewards=mm_rewards,
                                         z_mm=z_mm if pegasus else None,
                                         z_rr=z_rr if pegasus else None,
                                         mm_groups=mm_groups,
                                         **rollout_kwargs)
            # dims are timesteps x batch size x state/action/reward dims
            states, actions, rewards = trajectories
            if debug and i % 50 == 0:
                utils.plot_trajectories(*[
                    torch.stack(x).transpose(0, 1).detach().cpu().numpy()
                    for x in trajectories
                ])
            if callable(on_rollout):
                on_rollout(i, states, actions, rewards, discount)
        except RuntimeError:
            import traceback
            traceback.print_exc()
            print("RuntimeError")
            # resample random numbers
            resample()
            policy.zero_grad()
            dynamics.zero_grad()
            opt.zero_grad()
            continue

        # calculate loss. average over batch index, sum over time step index
        discounted_rewards = torch.stack(
            [r * discount(i) for i, r in enumerate(rewards)])
        if value_func is not None:
            Vend = value_func(states[-1], resample=False, return_samples=True)
            discounted_rewards = torch.cat(
                [discounted_rewards,
                 discount(H) * Vend.unsqueeze(0)], 0)
        if maximize:
            returns = -discounted_rewards.sum(0)
        else:
            returns = discounted_rewards.sum(0)

        if cvar_eps > -1.0 and cvar_eps < 1.0 and cvar_eps != 0:
            if cvar_eps > 0:
                # worst case optimizer
                q = np.quantile(returns.detach(), cvar_eps)
                returns = returns[returns.detach() < q]
            elif cvar_eps < 0:
                # best case optimizer
                q = np.quantile(returns.detach(), -cvar_eps)
                returns = returns[returns.detach() > q]

        if prioritized_replay and x0_idxs is not None:
            # apply importance sampling weights
            returns = returns * x0_weights

            # prepare hook to update priorities
            norms = []

            def accumulate_priorities(grad):
                norms.append(grad.norm(dim=-1))

            def update_priorities(x0_grad):
                global x0_tree
                norms.append(x0_grad.norm(dim=-1))
                # this contains the norms of gradients for every particle,
                # per time-step
                m_norms = torch.stack(norms)
                # group by initial state
                if mm_groups is not None:
                    m_norms = m_norms.view(-1, mm_groups,
                                           int(N_particles /
                                               mm_groups)).mean(-1)
                scores = m_norms.mean(0).detach().cpu().numpy(
                ) / x0_tree.counts[x0_idxs - x0_tree.max_size + 1]
                priorities = (scores + priority_eps)**priority_alpha
                # print(priorities.tolist())
                [x0_tree.update(idx, p) for idx, p in zip(x0_idxs, priorities)]
                x0_tree.renormalize()

            [
                actions[i].register_hook(accumulate_priorities)
                for i in range(1, len(actions))
            ]
            actions[0].register_hook(update_priorities)

        loss = returns.mean()

        # add regularization penalty
        if reg_weight > 0:
            loss = loss + reg_weight * policy.regularization_loss()

        # compute gradients
        loss.backward()

        if debug:
            for name, p in policy.named_parameters():
                print('{} p\tmean {},\tmin {},\tmax {},\tnorm {} '.format(
                    name, p.mean(), p.min(), p.max(), p.norm()))
            for name, p in policy.named_parameters():
                g = p.grad
                print('{} g\tmean {},\tmin {},\tmax {},\tnorm {} '.format(
                    name, g.mean(), g.min(), g.max(), g.norm()))

        # clip gradients
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)

        # update parameters
        opt.step()
        n_opt_steps += 1
        pbar.set_description((msg % (torch.stack(rewards).sum(0).mean())) +
                             ' [{0}]'.format(len(rewards)))

        if callable(on_iteration):
            on_iteration(i, loss, states, actions, rewards, discount)

        # sample initial states
        if exp is not None:
            if prioritized_replay:
                # first check if exp is bigger than x0_tree
                if exp.n_samples() > x0_tree.size:
                    # add states to sum tree
                    for idx in range(episode_counter, exp.n_episodes()):
                        for x in torch.tensor(exp.states[idx]):
                            x0_tree.append(x, x0_tree.max_p)
                            x0_tree.renormalize()
                    episode_counter = exp.n_episodes()

                if mm_groups is not None:
                    x0, x0_idxs, x0_weights = x0_tree.sample(
                        mm_groups, beta=priority_beta)
                else:
                    x0, x0_idxs, x0_weights = x0_tree.sample(
                        N_particles, beta=priority_beta)

                priority_beta = max(1.0,
                                    priority_beta + priority_beta_increase)
                x0 = torch.stack(x0).to(dynamics.X.device, dynamics.X.dtype)
                x0_weights = torch.tensor(np.stack(x0_weights)).to(
                    x0.device, x0.dtype)
                # print((x0_tree.counts == 1).sum(), x0_tree.counts.max())
                x0.requires_grad_(True)
            else:
                if mm_groups is not None:
                    # split the initial states into groups from
                    # different timesteps
                    x0 = exp.sample_states(mm_groups,
                                           timestep=step_idx_to_sample).to(
                                               dynamics.X.device,
                                               dynamics.X.dtype)
                else:
                    x0 = exp.sample_states(N_particles,
                                           timestep=step_idx_to_sample).to(
                                               dynamics.X.device,
                                               dynamics.X.dtype)
                init_states = x0

        else:
            x0 = init_states.detach()

    policy.eval()
    dynamics.eval()
    policy_update_counter[policy] = n_opt_steps


class MCPILCOAgent(torch.nn.Module):
    '''
    Utility class for instantiating an MCPILCO learning agent
    '''
    def __init__(self,
                 policy,
                 dynamics,
                 dataset,
                 pol_optimizer=None,
                 dyn_optimizer=None):
        super(MCPILCOAgent, self).__init__()
        self.pol = policy
        self.pol_optimizer = pol_optimizer
        self.dyn = dynamics
        self.dyn_optimizer = dyn_optimizer
        self.exp = dataset
        self.policy_update_counter = 0

    def sample_initial_states(self,
                              batch_size,
                              step_idx_to_sample=None,
                              init_state_noise=0.0):
        x0 = self.exp.sample_states(batch_size, timestep=step_idx_to_sample)
        x0 = x0.to(self.dyn.X.device, self.dyn.X.dtype)
        x0 += init_state_noise * torch.randn_like(x0)
        return x0

    def train(self,
              steps,
              batch_size=100,
              opt_iters=1000,
              pegasus=True,
              mm_states=False,
              mm_rewards=False,
              maximize=True,
              clip_grad=1.0,
              cvar_eps=0.0,
              reg_weight=0.0,
              discount=None,
              on_iteration=None,
              step_idx_to_sample=None,
              init_state_noise=0.0,
              resampling_period=500,
              debug=False):
        dynamics, policy, exp, opt = (self.dyn, self.pol, self.exp,
                                      self.pol_optimizer)
        dynamics.eval()
        policy.train()
        # init optimizer
        if opt is None:
            params = filter(lambda p: p.requires_grad, self.pol.parameters())
            opt = torch.optim.Adam(params)

        # init function for computing discount
        if discount is None:
            discount = lambda i: 1.0 / steps  # noqa: E731
        elif not callable(discount):
            discount_factor = discount
            discount = lambda i: discount_factor**i  # noqa: E731

        # init progress bar
        msg = ("Pred. Cumm. rewards: %f"
               if maximize else "Pred. Cumm. costs: %f")
        pbar = tqdm.tqdm(range(opt_iters), total=opt_iters)

        # init random numbers
        init_states = self.sample_initial_states(batch_size,
                                                 step_idx_to_sample,
                                                 init_state_noise)
        D = init_states.shape[-1]
        shape = init_states.shape
        dtype = init_states.dtype
        device = init_states.device
        z_mm = torch.randn(steps + shape[0],
                           *shape[1:]).reshape(-1, D).to(device, dtype)
        z_rr = torch.randn(steps + shape[0], 1).reshape(-1,
                                                        1).to(device, dtype)

        # sample initial random numbers
        def resample():
            self.dyn.resample()
            self.pol.resample()
            z_mm.normal_()
            z_rr.normal_()

        resample()

        for i in pbar:
            # zero gradients
            self.pol.zero_grad()
            self.dyn.zero_grad()
            opt.zero_grad()
            if (not pegasus
                    or self.policy_update_counter % resampling_period == 0):
                resample()

            # rollout policy
            try:
                trajectories = utils.rollout(init_states,
                                             self.dyn,
                                             self.pol,
                                             steps,
                                             resample_state_noise=True,
                                             resample_action_noise=True,
                                             mm_states=mm_states,
                                             mm_rewards=mm_rewards,
                                             z_mm=z_mm if pegasus else None,
                                             z_rr=z_rr if pegasus else None)
                # dims are timesteps x batch size x state/action/reward dims
                states, actions, rewards = trajectories
                if debug and i % 100 == 0:
                    utils.plot_trajectories(
                        states.transpose(0, 1).cpu().detach().numpy(),
                        actions.transpose(0, 1).cpu().detach().numpy(),
                        rewards.transpose(0, 1).cpu().detach().numpy())
            except RuntimeError:
                import traceback
                traceback.print_exc()
                print("RuntimeError")
                # resample random numbers
                resample()
                init_states = self.sample_initial_states(
                    batch_size, step_idx_to_sample, init_state_noise)
                continue

            # calculate loss. average over batch index, sum over time step
            # index
            discounted_rewards = torch.stack(
                [r * discount(i) for i, r in enumerate(rewards)])

            if maximize:
                returns = -discounted_rewards.sum(0)
            else:
                returns = discounted_rewards.sum(0)

            if cvar_eps > -1.0 and cvar_eps < 1.0 and cvar_eps != 0:
                if cvar_eps > 0:
                    # worst case optimizer
                    q = np.quantile(returns.detach(), cvar_eps)
                    loss = returns[returns.detach() < q].mean()
                elif cvar_eps < 0:
                    # best case optimizer
                    q = np.quantile(returns.detach(), -cvar_eps)
                    loss = returns[returns.detach() > q].mean()
            else:
                loss = returns.mean()

            # add regularization penalty
            if reg_weight > 0:
                loss = loss + reg_weight * policy.regularization_loss()

            # compute gradients
            loss.backward()

            # clip gradients
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)

            # update parameters
            opt.step()
            self.policy_update_counter += 1
            pbar.set_description((msg % (rewards.sum(0).mean())) +
                                 ' [{0}]'.format(len(rewards)))

            if callable(on_iteration):
                on_iteration(i, loss, states, actions, rewards, discount)

            # sample initial states
            N_particles = init_states.shape[0]
            x0 = exp.sample_states(N_particles,
                                   timestep=step_idx_to_sample).to(
                                       dynamics.X.device, dynamics.X.dtype)
            x0 += init_state_noise * torch.randn_like(x0)
            init_states = x0
            x0 = x0.detach()

        policy.eval()
        dynamics.eval()

    def fit_dynamics(self):
        '''
        '''
    def forward(self, x):
        '''
        Calling the agent is equivalent to evaluating its policy
        '''
        return self.policy(x)
