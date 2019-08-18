import torch

jit_scripts = {}


def mm_resample_infer_ns_(samples, z, jitter):
    M = samples.shape[0]
    m = samples.mean(0)
    deltas = samples - m
    S = deltas.t().mm(deltas) / (M - 1) + jitter
    L = S.cholesky()
    z = torch.mm(deltas, L.t().inverse()).detach()
    z = z.detach()
    return m + z.mm(L.t())


def mm_resample_(samples, z, jitter):
    M = samples.shape[0]
    m = samples.mean(0)
    deltas = samples - m
    S = deltas.t().mm(deltas) / (M - 1) + jitter
    L = S.cholesky()
    # make sure we don't underestimate the uncertainty
    z = (z - z.mean(0)) / z.std(0)
    z = z.detach()
    return m + z.mm(L.t())


def get_mm_resample_script(samples, z, jitter, infer_noise_variables):
    global jit_scripts
    inputs = (samples, z, jitter)
    key = (str(inp.type()) + '_' + str(inp.device) for inp in inputs)
    key = '_'.join(key) + str(infer_noise_variables)
    if key not in jit_scripts:
        mm_resample = (mm_resample_infer_ns_
                       if infer_noise_variables else mm_resample_)

        jit_scripts[key] = torch.jit.trace(mm_resample, inputs)
    return jit_scripts[key]


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
            breaking_condition=None,
            on_step=None,
            **kwargs):
    '''
        Obtains trajectory distribution (s_0, a_0, r_0, s_1, a_1, r_1,...)
        by rolling out the policy on the model, from the given set of states
    '''
    trajectory = []
    state_noise = torch.zeros_like(states)
    jitter1, jitter2 = None, None

    #mm_resample = (mm_resample_infer_ns_
    #               if infer_noise_variables else mm_resample_)
    mm_resample = get_mm_resample_script(states, torch.randn_like(states),
                                         torch.eye(states.shape[-1]),
                                         infer_noise_variables)

    for i in range(steps):
        try:
            # sample (or query) random numbers
            z1 = get_z_rnd(z_mm, i, states.shape, states.device)
            z2 = get_z_rnd(z_rr, i, (states.shape[0], 1), states.device)

            # noisy state measurement
            states_ = states + state_noise

            # evaluate policy
            actions = policy(states_,
                             resample=resample_policy,
                             output_noise=True,
                             return_samples=True,
                             resample_output_noise=resample_action_noise)

            # propagate state particles (and obtain rewards) TODO: make this an env.step call
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
                if jitter1 is None:
                    jitter1 = 1e-12 * torch.eye(states.shape[-1],
                                                device=states.device)
                next_states = mm_resample(next_states, z1, jitter1)

            # moment matching for rewards
            if mm_rewards:
                if jitter2 is None:
                    jitter2 = 1e-12 * torch.eye(rewards.shape[-1],
                                                device=rewards.device)
                rewards = mm_resample(rewards, z2, jitter2)

            # noisy reward measurements
            # rewards = rewards + 0.1 * reward_noise

            trajectory.append((states, actions, rewards))
            states = next_states
            if callable(breaking_condition):
                if breaking_condition(trajectory):
                    break
            if callable(on_step):
                on_step(trajectory)
        except RuntimeError as e:
            if len(trajectory) > 5:
                break
            raise e
    trajectory = [torch.stack(x) for x in zip(*trajectory)]
    return trajectory


def rollout_with_Qvalues(states,
                         dynamics,
                         policy,
                         Q,
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
                         breaking_condition=None,
                         on_step=None,
                         **kwargs):
    def compute_values(trajectory):
        # get last step
        states, actions, rewards = trajectory[-1]
        # compute and append values for last step
        values = Q(torch.cat([states, actions], -1))
        trajectory[-1].append(values)
        if callable(on_step):
            on_step(trajectory)
