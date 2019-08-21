import atexit
import datetime
import numpy as np
import os
import torch
import tensorboardX

from prob_mbrl import utils, algorithms, envs
torch.set_flush_denormal(True)
torch.set_num_threads(1)

if __name__ == '__main__':
    # parameters
    n_rnd = 10
    H = 40
    N_particles = 100
    dyn_components = 1
    dyn_hidden = [200] * 2
    use_cuda = True
    learn_reward = False

    # initialize environment
    env = envs.Cartpole()
    results_filename = os.path.expanduser(
        "~/.prob_mbrl/results_%s_%s.pth.tar" %
        (env.__class__.__name__,
         datetime.datetime.now().strftime("%Y%m%d%H%M%S.%f")))
    env.dt = env.model.dt
    D = env.observation_space.shape[0]
    U = env.action_space.shape[0]
    maxU = env.action_space.high

    # initialize reward/cost function
    if learn_reward or env.reward_func is None:
        reward_func = None
    else:
        reward_func = env.reward_func

    # initialize learning algorithm
    agent = algorithms.MBDDPG.MBDDPG(D,
                                     U,
                                     maxU,
                                     reward_func=reward_func,
                                     dyn_components=dyn_components,
                                     dyn_hidden=dyn_hidden)

    # initalize experience dataset
    exp = utils.ExperienceDataset()

    if use_cuda and torch.cuda.is_available():
        agent = agent.cuda()

    writer = tensorboardX.SummaryWriter()

    # callbacks
    def on_close():
        writer.close()

    atexit.register(on_close)

    # policy learning loop
    pol = agent
    for it in range(100 + n_rnd):
        if it < n_rnd:
            pol_ = (lambda x, t: maxU * (2 * np.random.rand(U, ) - 1)
                    )  # noqa: E731
        else:
            pol_ = pol

        # apply policy
        ret = utils.apply_controller(
            env, pol_, H, callback=lambda *args, **kwargs: env.render())
        params_ = [] if it < n_rnd else [
            p.clone() for p in list(pol.parameters())
        ]
        exp.append_episode(*ret, policy_params=params_)
        exp.save(results_filename)

        if it < n_rnd - 1:
            continue
        ps_it = it - n_rnd + 1

        def on_iteration(i, loss, states, actions, rewards, discount):
            writer.add_scalar('mc_pilco/episode_%d/training loss' % ps_it,
                              loss, i)
            if i % 100 == 0:
                states = states.transpose(0, 1).cpu().detach().numpy()
                actions = actions.transpose(0, 1).cpu().detach().numpy()
                rewards = rewards.transpose(0, 1).cpu().detach().numpy()
                utils.plot_trajectories(states,
                                        actions,
                                        rewards,
                                        plot_samples=False)

        # train agent
        agent.fit(exp, H, 120, batch_size=N_particles)

        # plot rollout
        x0 = torch.tensor(exp.sample_states(N_particles, timestep=0)).to(
            agent.dyn.X.device).float()
        x0 = x0 + 1e-1 * x0.std(0) * torch.randn_like(x0)
        x0 = x0.detach()
        utils.plot_rollout(x0, agent.dyn, agent.actor_target, H)
        writer.add_scalar('robot/evaluation_loss',
                          torch.tensor(ret[2]).sum(), ps_it + 1)
