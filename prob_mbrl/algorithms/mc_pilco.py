import torch
import tqdm

from prob_mbrl.utils import rollout, plot_trajectories


def mc_pilco(init_states,
             dynamics,
             policy,
             steps,
             opt=None,
             exp=None,
             opt_iters=1000,
             pegasus=True,
             mm_states=False,
             mm_rewards=False,
             maximize=True,
             clip_grad=1.0,
             discount=None,
             on_iteration=None,
             debug=False):
    dynamics.eval()
    policy.train()

    if discount is None:
        discount = lambda i: 1.0 / steps  # noqa: E731
    elif not callable(discount):
        discount_factor = discount
        discount = lambda i: discount_factor**i  # noqa: E731

    msg = "Cumm. rewards: %f" if maximize else "Cumm. costs: %f"
    if opt is None:
        params = filter(lambda p: p.requires_grad, policy.parameters())
        opt = torch.optim.Adam(params)
    pbar = tqdm.tqdm(range(opt_iters), total=opt_iters)
    D = init_states.shape[-1]
    shape = init_states.shape
    z_mm = torch.randn(steps + shape[0], *shape[1:])
    z_mm = z_mm.reshape(-1, D).float().to(dynamics.X.device)
    z_rr = torch.randn(steps + shape[0], 1)
    z_rr = z_rr.reshape(-1, 1).float().to(dynamics.X.device)

    def resample():
        dynamics.resample()
        policy.resample()
        z_mm.normal_()
        z_rr.normal_()

    # sample initial random numbers
    resample()

    x0 = init_states
    states = [init_states] * 2
    dynamics.eval()
    policy.train()

    for i in pbar:
        # zero gradients
        policy.zero_grad()
        dynamics.zero_grad()
        opt.zero_grad()
        if not pegasus or i % (opt_iters / 2) == 1:
            resample()

        # rollout policy
        H = steps
        try:
            trajs = rollout(
                x0,
                dynamics,
                policy,
                H,
                resample_state_noise=not pegasus,
                resample_action_noise=not pegasus,
                mm_states=mm_states,
                mm_rewards=mm_rewards,
                z_mm=z_mm if pegasus else None,
                z_rr=z_rr if pegasus else None)
            states, actions, rewards = (torch.stack(x) for x in zip(*trajs))
            if debug and i % 100 == 0:
                plot_trajectories(
                    states.transpose(0, 1).cpu().detach().numpy(),
                    actions.transpose(0, 1).cpu().detach().numpy(),
                    rewards.transpose(0, 1).cpu().detach().numpy())
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
        if maximize:
            loss = -discounted_rewards.sum(0).mean()
        else:
            loss = discounted_rewards.sum(0).mean()
        # add regularization penalty
        #loss = loss + 1e-3 * policy.regularization_loss()

        # compute gradients
        loss.backward()

        # clip gradients
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)

        # update parameters
        opt.step()
        if maximize:
            pbar.set_description((msg % (-loss)) +
                                 ' [{0}]'.format(len(rewards)))
        else:
            pbar.set_description((msg % (loss)) +
                                 ' [{0}]'.format(len(rewards)))

        if callable(on_iteration):
            on_iteration(i, loss, states, actions, rewards, opt, policy,
                         dynamics)

        # sample initial states
        if exp is not None:
            N_particles = init_states.shape[0]
            x0 = torch.tensor(exp.sample_states(N_particles)).to(
                dynamics.X.device).float()
            x0 += 1e-1 * init_states.std(0) * torch.randn_like(x0)
        else:
            x0 = init_states
        x0 = x0.detach()


class MCPILCOAgent(torch.nn.Module):
    '''
    Utility class for instantiating an MCPILCO learning agent
    '''

    def __init__(self,
                 policy=None,
                 dynmodel=None,
                 reward_func=None,
                 dataset=None):
        super(MCPILCOAgent, self).__init__()
        self.dataset = dataset
        self.pol = policy
        self.dyn = dynmodel

    def fit_policy(self,
                   init_states,
                   steps,
                   opt=None,
                   exp=None,
                   opt_iters=1000,
                   pegasus=True,
                   mm_states=False,
                   mm_rewards=False,
                   maximize=True,
                   clip_grad=1.0,
                   mpc=False,
                   max_steps=None,
                   on_iteration=None):
        '''
            Runs the MCPILCO loop
        '''
        dynamics = self.dyn
        policy = self.pol
        msg = ("Cumm. rewards: %f" if maximize else "Cumm. costs: %f")
        if opt is None:
            params = filter(lambda p: p.requires_grad, policy.parameters())
            opt = torch.optim.Adam(params)
        pbar = tqdm.tqdm(range(opt_iters), total=opt_iters)
        max_steps = steps if max_steps is None else max_steps
        D = init_states.shape[-1]
        shape = init_states.shape
        z_mm = None
        z_rr = None
        if pegasus:
            # sample initial random numbers
            z_mm = torch.randn(steps + shape[0], *shape[1:])
            z_mm = z_mm.reshape(-1, D).float().to(dynamics.X.device)
            z_rr = torch.randn(steps + shape[0], 1)
            z_rr = z_rr.reshape(-1, 1).float().to(dynamics.X.device)
            dynamics.resample()
            policy.resample()

        init_timestep = 0
        x0 = init_states
        states = [init_states] * 2
        sample_idx = torch.tensor(1).random_(0, x0.shape[0])
        dynamics.eval()
        policy.train()
        policy.zero_grad()
        dynamics.zero_grad()

        for i in pbar:
            if mpc:
                if init_timestep != 0:
                    # start from a sample from next simulated timestep
                    x0 = states[1].detach()
                    sample_idx.random_(x0.shape[0])
                    x0 = x0[sample_idx] * torch.ones_like(x0)
                    # add noise
                    x0 += init_states.std(0) * torch.randn_like(x0)

                init_timestep = (init_timestep + 1) % steps

            # rollout policy
            H = max_steps if mpc and init_timestep != 1 else steps
            n_retries = 4
            retries = 0
            while retries < n_retries:
                try:
                    trajs = rollout(
                        x0,
                        dynamics,
                        policy,
                        H,
                        resample_state_noise=not pegasus,
                        resample_action_noise=not pegasus,
                        mm_states=mm_states,
                        mm_rewards=mm_rewards,
                        z_mm=z_mm,
                        z_rr=z_rr)
                    break
                except RuntimeError:
                    # resample random numbers
                    dynamics.resample()
                    policy.resample()
                    retries += 1
            states, actions, rewards = (torch.stack(x) for x in zip(*trajs))

            # calculate loss. average over batch index, sum over time
            # step index
            if maximize:
                loss = -rewards.sum(0).mean()
            else:
                loss = rewards.sum(0).mean()

            if init_timestep == mpc * 1:
                loss0 = loss
            # compute gradients
            loss.backward()

            if init_timestep == 0:
                # clip gradients
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                                   clip_grad)

                # update parameters
                opt.step()
                pbar.set_description((msg % (loss0)) +
                                     ' [{0}]'.format(len(rewards)))

                if callable(on_iteration):
                    on_iteration(i, loss, states, actions, rewards, opt,
                                 policy, dynamics)

                # zero gradients
                policy.zero_grad()
                dynamics.zero_grad()

                # setup dynamics and policy
                if not pegasus:
                    dynamics.resample()
                    policy.resample()

                # sample initial states
                if exp is not None:
                    N_particles = init_states.shape[0]
                    x0 = torch.tensor(exp.sample_states(N_particles)).to(
                        dynamics.X.device).float()
                    x0 += 1e-1 * init_states.std(0) * torch.randn_like(x0)
                else:
                    x0 = init_states

    def fit_dynamics(self):
        '''
        '''

    def forward(self, x):
        '''
        Calling the agent is equivalent to evaluating its policy
        '''
        return self.policy(x)