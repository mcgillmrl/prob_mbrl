import numpy as np
import torch
import tqdm


def get_z_rnd(z, i, shape, device=None):
    if z is not None:
        idxs = torch.range(i, i+shape[0]-1).to(device).long()
        idxs %= shape[0]
        return z[idxs]
    else:
        return torch.randn(*shape, device=device)


def rollout(states, forward, policy, steps,
            resample_model=False,
            resample_policy=False,
            mm_states=False, mm_rewards=False,
            z_mm=None, z_states=None, z_rewards=None,
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
        z2 = get_z_rnd(z_states, i, states.shape, states.device)
        z3 = get_z_rnd(z_rewards, i, [states.shape[0], 1], states.device)

        # evaluate policy
        actions = policy(states, resample=resample_policy)

        # propagate state particles (and obtain rewards)
        next_states, rewards = forward(
            states, actions, measurement_noise=True,
            resample=resample_model, z_states=z2, z_rewards=z3)

        if mm_states:
            m = next_states.mean(0)
            deltas = next_states - m
            S = deltas.t().mm(deltas)/M + 1e-6*torch.eye(m.shape[-1])
            next_states = m + z1.mm(S.potrf())

        if mm_rewards:
            m = rewards.mean(0)
            deltas = rewards - m
            S = deltas.t().mm(deltas)/M + 1e-6*torch.eye(m.shape[-1])
            rewards = m + z3.mm(S.potrf())
            
        trajectory.append((states, actions, rewards))
        states = next_states
    return trajectory


def mc_pilco(init_states, forward, dynamics, policy, steps, opt=None, exp=None,
             opt_iters=1000, pegasus=True, mm_states=False, mm_rewards=False,
             maximize=True, clip_grad=1.0, mpc=False, max_steps=None):
    msg = "Accumulated rewards: %f" if maximize else "Accumulated costs: %f"
    if opt is None:
        params = filter(lambda p: p.requires_grad, policy.parameters())
        opt = torch.optim.Adam(params)
    pbar = tqdm.tqdm(range(opt_iters), total=opt_iters)
    max_steps = steps if max_steps is None else max_steps
    D = init_states.shape[-1]
    shape = init_states.shape
    z = {}
    if pegasus:
        # sample initial random numbers
        z['z_mm'] = torch.randn(
            steps+shape[0], *shape[1:]).reshape(-1, D).to(dynamics.X.device).float()
        z['z_states'] = torch.randn(
            steps+shape[0], *shape[1:]).reshape(-1, D).to(dynamics.X.device).float()
        z['z_rewards'] = torch.randn(
            steps+shape[0], 1).to(dynamics.X.device).float()
        dynamics.model.resample()
        policy.model.resample()

    init_timestep = 0
    x0 = init_states
    sample_idx = torch.tensor(1).random_(0, x0.shape[0])
    policy.zero_grad()
    dynamics.zero_grad()

    for i in pbar:
        if mpc:
            if init_timestep != 0:
                # start from a sample from next simulated timestep
                x0 = states[1].detach()
                sample_idx.random_(x0.shape[0])
                x0 = x0[sample_idx]*torch.ones_like(x0)
                # add noise
                x0 += init_states.std(0)*torch.randn_like(
                    x0)

            init_timestep = (init_timestep + 1) % steps

        # rollout policy
        H = max_steps if mpc and init_timestep != 1 else steps
        trajs = rollout(x0, forward, policy, H,
                        mm_states=mm_states, mm_rewards=mm_rewards, **z)
        states, actions, rewards = (torch.stack(x) for x in zip(*trajs))

        # calculate loss. average over batch index, sum over time step index
        loss = -rewards.mean(1).mean(0) if maximize else rewards.mean(1).mean(0)
        if init_timestep == mpc*1:
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
        
            # zero gradients
            policy.zero_grad()
            dynamics.zero_grad()

            # setup dynamics and policy
            if not pegasus:
                dynamics.model.resample()
                policy.model.resample()

            # sample initial states
            if exp is not None:
                N_particles = init_states.shape[0]
                x0 = torch.tensor(
                    exp.sample_states(N_particles)
                    ).to(dynamics.X.device).float()
                x0 += 1e-2*init_states.std(0)*torch.randn_like(
                    x0)
            else:
                x0 = init_states
