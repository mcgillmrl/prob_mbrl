import torch

jit_scripts = {}


def mm_resample_infer_ns_(samples, z, jitter):
    M = samples.shape[-2]
    m = samples.mean(-2, keepdim=True)
    deltas = samples - m
    deltasT = deltas.transpose(-1, -2)
    S = deltasT.matmul(deltas) / (M - 1) + jitter
    L = S.cholesky()
    LT = L.transpose(-1, -2)
    z = torch.triangular_solve(deltasT, L,
                               upper=False)[0].transpose(-1, -2).detach()
    #z = (deltas.matmul(LT.inverse())).detach()
    return m + z.matmul(LT)


def mm_resample_(samples, z, jitter):
    M = samples.shape[-2]
    m = samples.mean(-2, keepdim=True)
    deltas = samples - m
    S = deltas.transpose(-1, -2).matmul(deltas) / (M - 1) + jitter
    L = S.cholesky()
    # make sure we don't underestimate the uncertainty
    z = (z - z.mean(-2, keepdim=True)) / z.std(-2, keepdim=True)
    z = z.detach()
    return m + z.matmul(L.transpose(-1, -2))


def get_mm_resample_script(samples,
                           z,
                           jitter,
                           infer_noise_variables,
                           mm_groups=None):
    global jit_scripts
    if mm_groups is not None:
        inputs = (samples.view(mm_groups, -1, samples.shape[-1]),
                  z.view(mm_groups, -1, z.shape[-1]), jitter)
    else:
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
            mm_groups=None,
            breaking_condition=None,
            on_step=None,
            on_pol_eval=None,
            **kwargs):
    '''
        Obtains trajectory distribution (s_0, a_0, r_0, s_1, a_1, r_1,...)
        by rolling out the policy on the model, from the given set of states
    '''
    trajectory = []
    jitter1, jitter2 = None, None

    mm_resample = (mm_resample_infer_ns_
                   if infer_noise_variables else mm_resample_)
    # mm_resample = get_mm_resample_script(states, torch.randn_like(states),
    #                                      torch.eye(states.shape[-1]),
    #                                      infer_noise_variables, mm_groups)

    for i in range(steps):
        try:
            # sample (or query) random numbers
            z1 = get_z_rnd(z_mm, i, states.shape, states.device)
            z2 = get_z_rnd(z_rr, i, (states.shape[0], 1), states.device)

            # noisy state measurement
            states_ = states + state_noise  #   .detach()

            # evaluate policy
            actions = policy(states,
                             resample=resample_policy,
                             return_samples=True,
                             resample_noise=resample_action_noise)
            if callable(on_pol_eval):
                states, actions = on_pol_eval(i, states, actions)

            # propagate state particles (and obtain rewards)
            # TODO: make this an env.step call
            outs = dynamics((states, actions),
                            return_samples=True,
                            separate_outputs=True,
                            deltas=False,
                            resample=resample_model,
                            resample_noise=resample_state_noise)
            next_states, rewards = outs

            # moment matching for states
            if mm_states:
                if jitter1 is None:
                    jitter1 = 1e-12 * torch.eye(states.shape[-1],
                                                device=states.device)
                if mm_groups is not None:
                    next_states = mm_resample(
                        next_states.view(mm_groups, -1, next_states.shape[-1]),
                        z1.view(mm_groups, -1, z1.shape[-1]),
                        jitter1).view(-1, next_states.shape[-1])

                else:
                    next_states = mm_resample(next_states, z1, jitter1)

            # moment matching for rewards
            if mm_rewards:
                if jitter2 is None:
                    jitter2 = 1e-12 * torch.eye(rewards.shape[-1],
                                                device=rewards.device)
                if mm_groups is not None:
                    rewards = mm_resample(
                        rewards.view(mm_groups, -1, rewards.shape[-1]),
                        z2.view(mm_groups, -1, z2.shape[-1]),
                        jitter2).view(-1, rewards.shape[-1])
                else:
                    rewards = mm_resample(rewards, z2, jitter2)

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
    trajectory = [list(x) for x in zip(*trajectory)]

    # append last state
    trajectory[0].append(next_states)

    return trajectory


def rollout_with_values(states,
                        dynamics,
                        policy,
                        steps,
                        V,
                        resample_model=False,
                        resample_policy=False,
                        resample_state_noise=True,
                        resample_action_noise=True,
                        mm_states=False,
                        mm_rewards=False,
                        infer_noise_variables=False,
                        z_mm=None,
                        z_rr=None,
                        mm_groups=None,
                        breaking_condition=None,
                        on_step=None,
                        on_pol_eval=None,
                        **kwargs):
    def compute_values(trajectory):
        # get last step
        states, actions, rewards = trajectory[-1]
        # compute and append values for last step
        values = V(states.detach(),
                   output_noise=True,
                   return_samples=True,
                   resample=resample_model,
                   resample_noise=resample_state_noise)
        trajectory[-1] = ((states, actions, rewards, values))
        if callable(on_step):
            on_step(trajectory)

    trajectory = rollout(states,
                         dynamics,
                         policy,
                         steps,
                         resample_model,
                         resample_policy,
                         resample_state_noise,
                         resample_action_noise,
                         mm_states,
                         mm_rewards,
                         infer_noise_variables,
                         z_mm,
                         z_rr,
                         mm_groups,
                         breaking_condition,
                         on_step=compute_values,
                         on_pol_eval=on_pol_eval)
    # append value of last state
    # get last step
    states = trajectory[0][-1]

    # compute and append values for last step
    values = V(states,
               output_noise=True,
               return_samples=True,
               resample=resample_model,
               resample_noise=resample_state_noise)
    trajectory[-1].append(values)

    return trajectory


def rollout_with_Qvalues(states,
                         dynamics,
                         policy,
                         steps,
                         Q,
                         resample_model=False,
                         resample_policy=False,
                         resample_state_noise=True,
                         resample_action_noise=True,
                         mm_states=False,
                         mm_rewards=False,
                         infer_noise_variables=False,
                         z_mm=None,
                         z_rr=None,
                         mm_groups=None,
                         breaking_condition=None,
                         on_step=None,
                         on_pol_eval=None,
                         **kwargs):
    def compute_values(trajectory):
        # get last step
        states, actions, rewards = trajectory[-1]
        # compute and append values for last step
        values = Q(torch.cat([states, actions].detach(), -1),
                   output_noise=True,
                   return_samples=True,
                   resample=resample_model,
                   resample_noise=resample_state_noise)
        trajectory[-1] = ((states, actions, rewards, values))
        if callable(on_step):
            on_step(trajectory)

    trajectory = rollout(states,
                         dynamics,
                         policy,
                         steps,
                         resample_model,
                         resample_policy,
                         resample_state_noise,
                         resample_action_noise,
                         mm_states,
                         mm_rewards,
                         infer_noise_variables,
                         z_mm,
                         z_rr,
                         mm_groups,
                         breaking_condition,
                         on_step=compute_values,
                         on_pol_eval=on_pol_eval)
    # append value of last state
    # get last step
    states = trajectory[0][-1]
    actions = policy(states,
                     resample=resample_policy,
                     output_noise=True,
                     return_samples=True,
                     resample_noise=resample_action_noise)
    # compute and append values for last step
    values = Q(torch.cat([states, actions], -1),
               output_noise=True,
               return_samples=True,
               resample=resample_model,
               resample_noise=resample_state_noise)
    trajectory[-1].append(values)

    return trajectory
