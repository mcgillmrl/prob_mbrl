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
            forward,
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
        next_states, rewards = forward(
            states,
            actions,
            output_noise=False,
            resample=resample_model,
            resample_output_noise=resample_state_noise)

        if mm_states:
            m = next_states.mean(0)
            deltas = next_states - m
            jitter = 1e-12 * torch.eye(m.shape[-1], device=m.device)
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
            jitter = 1e-12 * torch.eye(m.shape[-1], device=m.device)
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
             forward,
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
        trajs = rollout(
            x0,
            forward,
            policy,
            H,
            resample_model_noise=not pegasus,
            mm_states=mm_states,
            mm_rewards=mm_rewards,
            z_mm=z_mm,
            z_rr=z_rr)
        states, actions, rewards = (torch.stack(x) for x in zip(*trajs))

        # calculate loss. average over batch index, sum over time step index
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
                torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)

            # update parameters
            opt.step()
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
