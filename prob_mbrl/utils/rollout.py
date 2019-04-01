import torch


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
            resample_state_noise=True,
            resample_action_noise=True,
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
    state_noise = torch.zeros_like(states)
    for i in range(steps):
        # sample (or query) random numbers
        z1 = get_z_rnd(z_mm, i, states.shape, states.device)
        z2 = get_z_rnd(z_rr, i, (states.shape[0], 1), states.device)

        # evaluate policy
        actions = policy(
            states + state_noise,
            resample=resample_policy,
            resample_output_noise=resample_action_noise)

        # propagate state particles (and obtain rewards)
        outs = dynamics((states, actions),
                        output_noise=True,
                        return_samples=True,
                        separate_outputs=True,
                        deltas=False,
                        resample=resample_model,
                        resample_output_noise=resample_state_noise)
        next_states, rewards = outs[0]
        state_noise, reward_noise = outs[1]

        # moment matching for states
        if mm_states:
            m = next_states.mean(0)
            deltas = next_states - m
            jitter = 1e-12 * torch.eye(m.shape[-1], device=m.device)
            S = deltas.t().mm(deltas) / (M - 1) + jitter
            L = S.cholesky()
            if infer_noise_variables:
                z1 = torch.mm(deltas, L.inverse()).detach()
            else:
                # make sure we don't underestimate the uncertainty
                z1 = (z1 - z1.mean(0)) / z1.std(0)
            z1 = z1.detach()
            next_states = m + z1.mm(L)

        # moment matching for rewards
        if mm_rewards:
            m = rewards.mean(0)
            deltas = rewards - m
            jitter = 1e-12 * torch.eye(m.shape[-1], device=m.device)
            S = deltas.t().mm(deltas) / (M - 1) + jitter
            L = S.cholesky()
            if infer_noise_variables:
                z2 = torch.mm(deltas, L.inverse()).detach()
            else:
                # make sure we don't underestimate the uncertainty
                z2 = (z2 - z2.mean(0)) / z2.std(0)
            z2 = z2.detach()
            rewards = m + z2.mm(L)

        trajectory.append((states, actions, rewards))
        states = next_states
    return trajectory
