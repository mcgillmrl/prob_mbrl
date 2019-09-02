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

def update_value_function(V,
                          opt,
                          H,
                          i,
                          states,
                          actions,
                          rewards,
                          discount,
                          V_target=None,
                          reg_weight=1e-3,
                          resample=False,
                          polyak_averaging=0.005):
    V_tgt = V if V_target is None else V_target
    V.train()

    V.zero_grad()
    N = rewards[0].shape[0]
    discounted_rewards = torch.stack(
        [r * discount(j) for j, r in enumerate(rewards[:H])])
    returns = discounted_rewards.sum(0).detach()

    # when we evaluate the value function with resample=True, we use
    # the same seed. this ensures that we don't overwrite the noise masks
    # used when resample = False, but that we get the same masks for V0 and VH
    seed = torch.randint(2**32, [1])
    if V.output_density is None:
        V0 = V(states[0].detach(), resample=resample, seed=seed)
        VH = V_tgt(states[H].detach(), resample=resample, seed=seed)
        targets = returns + discount(H) * VH.detach()
        loss = torch.nn.functional.mse_loss(V0, targets)
    else:
        # the output of the network are the parameters of a probability density
        pV0 = V(states[0].detach(),
                resample=resample,
                seed=seed,
                return_samples=False)
        VH = V_tgt(states[H].detach(),
                   resample=resample,
                   seed=seed,
                   return_samples=True,
                   output_noise=False)
        targets = returns + discount(H) * VH.detach()
        loss = V.output_density.log_prob(targets, *pV0).mean()

    if hasattr(V, 'regularization_loss'):
        loss += reg_weight * V.regularization_loss()
    loss.backward()

    opt.step()

    if V_target is not None and polyak_averaging > 0:
        tau = polyak_averaging
        for param, target_param in zip(V.parameters(), V_target.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)
    V.eval()
    #print(torch.cat([rewards.sum(0), returns, targets, pV0[0]], -1))


def update_Qvalue_function(Q,
                           policy,
                           opt,
                           H,
                           i,
                           states,
                           actions,
                           rewards,
                           discount,
                           Q_target=None,
                           polyak_averaging=0.005):
    Q_tgt = Q if Q_target is None else Q_target
    Q.zero_grad()
    N = rewards[0].shape[0]
    discounted_rewards = torch.stack(
        [r * discount(j) for j, r in enumerate(rewards[:H])])
    returns = discounted_rewards.sum(0).detach()

    # we evaluate the value function with resample=True, but with the same seed
    # this ensures that we don't overwrite the noise masks used when
    # resample = False, but that we get the same masks for V0 and VH
    seed = torch.randint(2**32, [1])
    inps0 = torch.cat([states[0], actions[0]], -1)
    inpsH = torch.cat([states[H], policy(states[H], resample=True)], -1)
    if Q.output_density is None:
        Q0 = Q(inps0.detach(), resample=True, seed=seed)
        QH = Q_tgt(inpsH.detach(), resample=True, seed=seed)
        targets = returns + discount(H) * QH.detach()
        loss = torch.nn.functional.mse_loss(Q0, targets)
    else:
        # the output of the network are the parameters of a probability density
        pQ0 = Q(inps0.detach(), resample=True, seed=seed, return_samples=False)
        QH = Q_tgt(inpsH.detach(),
                   resample=True,
                   seed=seed,
                   return_samples=True,
                   output_noise=False)
        targets = returns + discount(H) * QH.detach()
        loss = V.output_density.log_prob(targets, *pQ0).mean()

    if hasattr(V, 'regularization_loss'):
        loss += V.regularization_loss() / N
    loss.backward()

    opt.step()

    if Q_target is not None and polyak_averaging > 0:
        tau = polyak_averaging
        for param, target_param in zip(Q.parameters(), Q_target.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)
    #print(torch.cat([rewards.sum(0), returns, targets, pQ0[0]], -1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Deep-PILCO with moment matching")
    parser.add_argument('-e', '--env', type=str, default="Cartpole")
    parser.add_argument('-o',
                        '--output_folder',
                        type=str,
                        default="~/.prob_mbrl/")
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--n_initial_epi', type=int, default=0)
    parser.add_argument('--pred_H', type=int, default=25)
    parser.add_argument('--control_H', type=int, default=40)
    parser.add_argument('--discount_factor', type=float, default=None)

    parser.add_argument('--dyn_lr', type=float, default=1e-4)
    parser.add_argument('--dyn_opt_iters', type=int, default=2000)
    parser.add_argument('--dyn_batch_size', type=int, default=100)
    parser.add_argument('--dyn_drop_rate', type=float, default=0.1)
    parser.add_argument('--dyn_components', type=int, default=1)
    parser.add_argument('--dyn_shape',
                        type=lambda s: [int(d) for d in s.split(',')],
                        default=[200, 200])

    parser.add_argument('--pol_lr', type=float, default=1e-4)
    parser.add_argument('--pol_clip', type=float, default=1.0)
    parser.add_argument('--pol_drop_rate', type=float, default=0.1)
    parser.add_argument('--pol_opt_iters', type=int, default=1000)
    parser.add_argument('--pol_batch_size', type=int, default=100)
    parser.add_argument('--ps_iters', type=int, default=100)
    parser.add_argument('--pol_shape',
                        type=lambda s: [int(d) for d in s.split(',')],
                        default=[200, 200])

    parser.add_argument('--plot_level', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--learn_reward', action='store_true')
    parser.add_argument('--keep_best', action='store_true')
    parser.add_argument('--stop_when_done', action='store_true')

    # parameters
    args = parser.parse_args()
    locals().update(args.__dict__)

    # initialize environment
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.env in envs.__all__:
        env = envs.__dict__[args.env]()
    else:
        import gym
        env = gym.make(env.name)

    env_name = env.spec.id if env.spec is not None else env.__class__.__name__
    output_folder = os.path.expanduser(args.output_folder)
    results_filename = os.path.join(
        output_folder, "mc_pilco_no_mm_val/{}/experience_{}.pth.tar".format(
            env_name,
            datetime.datetime.now().strftime("%Y%m%d%H%M%S.%f")))
    D = env.observation_space.shape[0]
    U = env.action_space.shape[0]
    maxU = env.action_space.high
    minU = env.action_space.low

    # initialize reward/cost function
    if (args.learn_reward or not hasattr(env, 'reward_func')
            or env.reward_func is None):
        reward_func = None
        args.learn_reward = True
    else:
        reward_func = env.reward_func

    # intialize to max episode steps if available
    if hasattr(env, 'spec'):
        if hasattr(env.spec, 'max_episode_steps'):
            control_H = env.spec.max_episode_steps
            args.stop_when_done = True
    initial_experience = control_H * n_initial_epi

    # initialize dynamics model
    dynE = 2 * (D + 1) if args.learn_reward else 2 * D
    if dyn_components > 1:
        output_density = models.GaussianMixtureDensity(dynE / 2,
                                                       dyn_components)
        dynE = (dynE + 1) * dyn_components + 1
    else:
        output_density = models.DiagGaussianDensity(dynE / 2)

    dyn_model = models.mlp(
        D + U,
        dynE,
        dyn_shape,
        dropout_layers=[
            models.modules.CDropout(dyn_drop_rate * np.ones(hid))
            if dyn_drop_rate > 0 else None for hid in dyn_shape
        ],
        nonlin=torch.nn.ReLU)
    dyn = models.DynamicsModel(dyn_model,
                               reward_func=reward_func,
                               output_density=output_density).float()

    # initalize policy
    pol_model = models.mlp(D,
                           2 * U,
                           pol_shape,
                           dropout_layers=[
                               models.modules.BDropout(pol_drop_rate)
                               if pol_drop_rate > 0 else None
                               for hid in pol_shape
                           ],
                           biases_initializer=None,
                           nonlin=torch.nn.ReLU,
                           output_nonlin=partial(models.DiagGaussianDensity,
                                                 U))

    pol = models.Policy(pol_model, maxU, minU).float()

    # initialize value function approximator
    critic_model = models.mlp(D,
                              2,
                              val_hidden,
                              dropout_layers=[
                                  models.modules.CDropout(
                                      0.25 * np.random.rand(hid), 0.1)
                                  for hid in val_hidden
                              ],
                              nonlin=torch.nn.Tanh)
    V = models.Regressor(critic_model,
                         output_density=models.DiagGaussianDensity(1)).float()
    V_target = copy.deepcopy(V)
    V_target.load_state_dict(V.state_dict())

    print('Dynamics model\n', dyn)
    print('Policy\n', pol)
    print('Critic\n', V)

    # initalize experience dataset
    exp = utils.ExperienceDataset()

    # initialize dynamics optimizer
    opt1 = torch.optim.Adam(dyn.parameters(), dyn_lr)

    # initialize policy optimizer
    opt2 = torch.optim.Adam(pol.parameters(), pol_lr)

    # initialize critic optimizer
    opt3 = torch.optim.Adam(V.parameters(), 1e-4)

    if args.use_cuda and torch.cuda.is_available():
        dyn = dyn.cuda()
        pol = pol.cuda()

    writer = tensorboardX.SummaryWriter(logdir=os.path.join(
        output_folder, "mc_pilco_no_mm/{}/logs/".format(env_name)))

    # callbacks
    def on_close():
        writer.close()

    atexit.register(on_close)

    # initial experience data collection
    env.seed(seed)
    rnd = lambda x, t: env.action_space.sample()  # noqa: E731
    while exp.n_samples() < initial_experience:
        ret = utils.apply_controller(
            env,
            rnd,
            min(control_H, initial_experience - exp.n_samples() + 1),
            stop_when_done=args.stop_when_done)
        exp.append_episode(*ret, policy_params=copy.deepcopy(pol.state_dict()))
        exp.save(results_filename)

    # policy learning loop
    render_fn = (lambda *args, **kwargs: env.render()) if args.render else None
    update_V_fn = partial(
        update_value_function, V, opt3, pred_H)
    for ps_it in range(ps_iters):
        # apply policy
        new_exp = exp.n_samples() + control_H
        while exp.n_samples() < new_exp:
            ret = utils.apply_controller(env,
                                         pol,
                                         min(control_H,
                                             new_exp - exp.n_samples() + 1),
                                         stop_when_done=args.stop_when_done,
                                         callback=render_fn)
            exp.append_episode(*ret,
                               policy_params=copy.deepcopy(pol.state_dict()))
            exp.save(results_filename)

        # train dynamics
        X, Y = exp.get_dynmodel_dataset(deltas=True,
                                        return_costs=args.learn_reward)
        dyn.set_dataset(X.to(dyn.X.device, dyn.X.dtype),
                        Y.to(dyn.X.device, dyn.X.dtype))
        utils.train_regressor(dyn,
                              dyn_opt_iters,
                              dyn_batch_size,
                              True,
                              opt1,
                              log_likelihood=dyn.output_density.log_prob,
                              summary_writer=writer,
                              summary_scope='model_learning/episode_%d' %
                              ps_it)

        # sample initial states for policy optimization
        x0 = exp.sample_states(pol_batch_size,
                               timestep=0).to(dyn.X.device,
                                              dyn.X.dtype).detach()

        if args.plot_level > 0:
            utils.plot_rollout(x0[:25], dyn, pol, pred_H * 2)

        # train policy
        def on_iteration(i, loss, states, actions, rewards, discount):
            writer.add_scalar('mc_pilco/episode_%d/training loss' % ps_it,
                              loss, i)

        print("Policy search iteration %d" % (ps_it + 1))
        algorithms.mc_pilco(x0,
                            dyn,
                            pol,
                            pred_H,
                            opt2,
                            exp,
                            pol_opt_iters,
                            value_func=V,
                            discount=discount_factor,
                            pegasus=True,
                            mm_states=False,
                            mm_rewards=False,
                            maximize=True,
                            clip_grad=args.pol_clip,
                            step_idx_to_sample=None,
                            init_state_noise=1e-1 * x0.std(0),
                            prioritized_replay=True,
                            on_iteration=on_iteration,
                            on_rollout=update_V_fn)
        if args.plot_level > 0:
            utils.plot_rollout(x0[:25], dyn, pol, pred_H * 2)
        writer.add_scalar('robot/evaluation_loss',
                          torch.tensor(ret[2]).sum(), ps_it + 1)
