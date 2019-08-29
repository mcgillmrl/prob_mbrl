import argparse
import atexit
import copy
import datetime
import numpy as np
import os
import torch
import tensorboardX

from functools import partial
from prob_mbrl import utils, models, algorithms, envs
torch.set_flush_denormal(True)
torch.set_num_threads(2)


def perturb_initial_action(i, states, actions):
    if i == 0:
        actions = actions + 1e-1 * (torch.randint(0,
                                                  2,
                                                  actions.shape[0:],
                                                  device=actions.device,
                                                  dtype=actions.dtype) *
                                    actions.std(0)).detach()
    return states, actions


def threshold_linear(x, y0, yend, x0, xend):
    y = (x - x0) * (yend - y0) / (xend - x0) + y0
    return np.maximum(y0, np.minimum(yend, y)).astype(np.int32)


if __name__ == '__main__':
    # parameters
    seed = 0
    n_initial_epi = 1
    pred_H = partial(threshold_linear, y0=10, yend=25, x0=5, xend=20)
    control_H = 40
    discount_factor = None
    dyn_batch_size = 100
    pol_batch_size = 100
    N_polopt = 1000
    N_dynopt = 2000
    N_ps = 250
    dyn_components = 1
    dyn_hidden = [200] * 2
    pol_hidden = [64] * 2
    use_cuda = False
    learn_reward = True
    keep_best = False
    stop_when_done = False

    # initialize environment
    env = envs.Pendulum()
    #env = envs.Cartpole()
    import gym
    env = gym.make("HalfCheetah-v3")
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_name = env.spec.id if env.spec is not None else env.__class__.__name__
    results_filename = os.path.expanduser(
        "~/.prob_mbrl/mc_pilco_no_mm/%s_%s.pth.tar" %
        (env_name, datetime.datetime.now().strftime("%Y%m%d%H%M%S.%f")))
    D = env.observation_space.shape[0]

    U = env.action_space.shape[0]
    maxU = env.action_space.high
    minU = env.action_space.low

    # initialize reward/cost function
    if (learn_reward or not hasattr(env, 'reward_func')
            or env.reward_func is None):
        reward_func = None
    else:
        reward_func = env.reward_func

    # intialize to max episode steps if available
    if hasattr(env, 'spec'):
        if hasattr(env.spec, 'max_episode_steps'):
            control_H = env.spec.max_episode_steps
            stop_when_done = True
    initial_experience = control_H * n_initial_epi

    # initialize dynamics model
    dynE = 2 * (D + 1) if learn_reward else 2 * D
    if dyn_components > 1:
        output_density = models.GaussianMixtureDensity(dynE / 2,
                                                       dyn_components)
        dynE = (dynE + 1) * dyn_components + 1
    else:
        output_density = models.DiagGaussianDensity(dynE / 2)

    dyn_model = models.mlp(D + U,
                           dynE,
                           dyn_hidden,
                           dropout_layers=[
                               models.modules.CDropout(
                                   0.25 * np.random.rand(hid), 0.1)
                               for hid in dyn_hidden
                           ],
                           nonlin=torch.nn.ReLU)
    dyn = models.DynamicsModel(dyn_model,
                               reward_func=reward_func,
                               output_density=output_density).float()

    # initalize policy
    pol_model = models.mlp(
        D,
        2 * U,
        pol_hidden,
        dropout_layers=[
            models.modules.BDropout(0.25 * np.random.rand(hid))
            for hid in pol_hidden
        ],
        weights_initializer=partial(torch.nn.init.normal, std=1e-4),
        biases_initializer=None,
        nonlin=torch.nn.ReLU,
        output_nonlin=partial(models.DiagGaussianDensity, U))

    pol = models.Policy(pol_model, maxU, minU).float()

    print('Dynamics model\n', dyn)
    print('Policy\n', pol)

    # initalize experience dataset
    exp = utils.ExperienceDataset()

    # initialize dynamics optimizer
    opt1 = torch.optim.Adam(dyn.parameters(), 1e-4)

    # initialize policy optimizer
    opt2 = torch.optim.Adam(pol.parameters(), 1e-4)

    if use_cuda and torch.cuda.is_available():
        dyn = dyn.cuda()
        pol = pol.cuda()

    writer = tensorboardX.SummaryWriter()

    # callbacks
    def on_close():
        writer.close()

    atexit.register(on_close)

    # initial experience data collection
    env.seed(seed)
    scale = maxU - minU
    bias = minU
    rnd = lambda x, t: (scale * np.random.rand(U, ) + bias)  # noqa: E731
    while exp.n_samples() < initial_experience:
        ret = utils.apply_controller(
            env,
            rnd,
            min(control_H, initial_experience - exp.n_samples() + 1),
            stop_when_done=stop_when_done)
        exp.append_episode(*ret, policy_params=copy.deepcopy(pol.state_dict()))
        exp.save(results_filename)

    # policy learning loop
    for ps_it in range(N_ps):
        # apply policy
        new_exp = exp.n_samples() + control_H
        while exp.n_samples() < new_exp:
            ret = utils.apply_controller(env,
                                         pol,
                                         min(control_H,
                                             new_exp - exp.n_samples() + 1),
                                         callback=lambda *args: env.render(),
                                         stop_when_done=stop_when_done)
            exp.append_episode(*ret,
                               policy_params=copy.deepcopy(pol.state_dict()))
            exp.save(results_filename)

        # train dynamics
        X, Y = exp.get_dynmodel_dataset(deltas=True, return_costs=learn_reward)
        dyn.set_dataset(X.to(dyn.X.device).float(), Y.to(dyn.X.device).float())
        utils.train_regressor(dyn,
                              N_dynopt,
                              dyn_batch_size,
                              True,
                              opt1,
                              log_likelihood=dyn.output_density.log_prob,
                              reg_weight=max(1e-4, 1.0 / dyn.X.shape[0]),
                              summary_writer=writer,
                              summary_scope='model_learning/episode_%d' %
                              ps_it)

        # sample initial states for policy optimization
        x0 = exp.sample_states(pol_batch_size,
                               timestep=0).to(dyn.X.device).float().detach()

        utils.plot_rollout(x0[:25], dyn, pol, pred_H(ps_it) * 2)

        # train policy
        def on_iteration(i, loss, states, actions, rewards, discount):
            writer.add_scalar('mc_pilco/episode_%d/training loss' % ps_it,
                              loss, i)

        print("Policy search iteration %d" % (ps_it + 1))

        algorithms.mc_pilco(
            x0,
            dyn,
            pol,
            pred_H(ps_it),
            opt2,
            exp,
            N_polopt,
            discount=discount_factor,
            pegasus=True,
            mm_states=False,
            mm_rewards=False,
            maximize=True,
            clip_grad=1.0,
            init_state_noise=0.0,
            step_idx_to_sample=None,
            prioritized_replay=True,
            on_iteration=on_iteration,
            rollout_kwargs=dict(on_pol_eval=perturb_initial_action))
        utils.plot_rollout(x0[:25], dyn, pol, pred_H(ps_it) * 2)
        writer.add_scalar('robot/evaluation_loss',
                          torch.tensor(ret[2]).sum(), ps_it + 1)
