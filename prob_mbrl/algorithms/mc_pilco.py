import torch
import tqdm


def get_z_rnd(z, i, shape, device=None):
    if z is not None:
        idxs = torch.arange(i, i + shape[0]).to(device).long()
        idxs %= shape[0]
        return z[idxs]
    else:
        return torch.randn(*shape, device=device)


def rollout(states,
            dynamics,
            policy,
            steps,
            resample_model=False,
            resample_policy=False,
            resample_state_noise=False,
            resample_action_noise=False,
            mm_states=False,
            mm_rewards=False,
            infer_noise_variables=False,
            z_mm=None,
            z_rr=None,
            **kwargs):
    '''
        Obtains trajectory distribution (s_0, a_0, r_0, s_1, a_1, r_1,...)
        by rolling out the policy on the model, from the given set of states
    '''
    trajectory = []
    M = states.shape[0]
    for i in range(steps):
        # sample (or query) random numbers
        z1 = get_z_rnd(z_mm, i, states.shape, states.device)
        z2 = get_z_rnd(z_rr, i, (states.shape[0], 1), states.device)

        # evaluate policy
        actions = policy(
            states,
            resample=resample_policy,
            resample_output_noise=resample_action_noise)

        # propagate state particles (and obtain rewards)
        next_states, rewards = dynamics(
            (states, actions),
            output_noise=True,
            return_samples=True,
            separate_outputs=True,
            deltas=False,
            resample=resample_model,
            resample_output_noise=resample_state_noise)

        if mm_states:
            m = next_states.mean(0)
            deltas = next_states - m
            jitter = 1e-9 * torch.eye(m.shape[-1], device=m.device)
            S = deltas.t().mm(deltas) / (M - 1) + jitter
            L = S.potrf()
            if infer_noise_variables:
                z1 = torch.mm(deltas, L.inverse()).detach()
            else:
                # make sure we don't underestimate the uncertainty
                z1 = (z1 - z1.mean(0)) / z1.std(0)
            z1.requires_grad = False
            next_states = m + z1.mm(L)

        if mm_rewards:
            m = rewards.mean(0)
            deltas = rewards - m
            jitter = 1e-9 * torch.eye(m.shape[-1], device=m.device)
            S = deltas.t().mm(deltas) / (M - 1) + jitter
            L = S.potrf()
            if infer_noise_variables:
                z2 = torch.mm(deltas, L.inverse()).detach()
            else:
                # make sure we don't underestimate the uncertainty
                z2 = (z2 - z2.mean(0)) / z2.std(0)
            z2.requires_grad = False
            rewards = m + z2.mm(L)

        trajectory.append((states, actions, rewards))
        states = next_states
    return trajectory


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
             mpc=False,
             max_steps=None,
             on_iteration=None):
    msg = "Accumulated rewards: %f" if maximize else "Accumulated costs: %f"
    if opt is None:
        params = filter(lambda p: p.requires_grad, policy.parameters())
        opt = torch.optim.Adam(params)
    pbar = tqdm.tqdm(range(opt_iters), total=opt_iters)
    max_steps = steps if max_steps is None else max_steps
    D = init_states.shape[-1]
    shape = init_states.shape
    z_mm = None
    z_rr = None

    def resample():
        dynamics.resample()
        policy.resample()
        z_mm = torch.randn(steps + shape[0], *shape[1:])
        z_mm = z_mm.reshape(-1, D).float().to(dynamics.X.device)
        z_rr = torch.randn(steps + shape[0], 1)
        z_rr = z_rr.reshape(-1, 1).float().to(dynamics.X.device)

    if pegasus:
        # sample initial random numbers
        resample()

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
        try:
            trajs = rollout(
                x0,
                dynamics,
                policy,
                H,
                resample_model_noise=not pegasus,
                mm_states=mm_states,
                mm_rewards=mm_rewards,
                z_mm=z_mm,
                z_rr=z_rr)
            states, actions, rewards = (torch.stack(x) for x in zip(*trajs))
        except RuntimeError:
            # resample random numbers
            resample()
            continue

        # calculate loss. average over batch index, sum over time step index
        def discount(i):
            # return 0.99**i
            return 1.0 / len(rewards)

        discounted_rewards = torch.stack(
            [r * discount(i) for i, r in enumerate(rewards)])
        if maximize:
            loss = -discounted_rewards.sum(0).mean()
        else:
            loss = discounted_rewards.sum(0).mean()

        if init_timestep == mpc * 1:
            loss0 = loss
        # compute gradients
        loss.backward()

        if init_timestep == 0:
            # clip gradients
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)

            # update parameters
            opt.step()
            if maximize:
                pbar.set_description(msg % (-loss0))
            else:
                pbar.set_description(msg % (loss0))

            if callable(on_iteration):
                on_iteration(i, loss, states, actions, rewards, opt, policy,
                             dynamics)

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
        msg = ("Accumulated rewards: %f"
               if maximize else "Accumulated costs: %f")
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
                        resample_model_noise=not pegasus,
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
                pbar.set_description(msg % (loss0))

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