import atexit
import datetime
import numpy as np
import os
import torch
import tensorboardX

from prob_mbrl import utils, models, algorithms, losses, train_regressor, envs
torch.set_flush_denormal(True)
torch.set_num_threads(1)

if __name__ == '__main__':
    # parameters
    n_rnd = 10
    H = 60
    N_particles = 100
    dyn_components = 4
    dyn_hidden = [200] * 2
    pol_hidden = [50] * 2
    use_cuda = False
    learn_reward = False

    # initialize environment
    env = envs.DoubleCartpole()
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

    # initialize dynamics model
    dynE = 2 * (D + 1) if learn_reward else 2 * D
    if dyn_components > 1:
        output_density = models.MixtureDensity(dynE / 2, dyn_components)
        dynE = (dynE + 1) * dyn_components
        log_likelihood_loss = losses.gaussian_mixture_log_likelihood
    else:
        output_density = models.DiagGaussianDensity(dynE / 2)
        log_likelihood_loss = losses.gaussian_log_likelihood

    dyn_model = models.mlp(
        D + U,
        dynE,
        dyn_hidden,
        dropout_layers=[
            models.modules.CDropout(0.5, 0.1) for i in range(len(dyn_hidden))
        ],
        nonlin=torch.nn.ReLU)
    dyn = models.DynamicsModel(
        dyn_model, reward_func=reward_func,
        output_density=output_density).float()

    # initalize policy
    pol_model = models.mlp(
        D,
        U,
        pol_hidden,
        dropout_layers=[
            models.modules.BDropout(0.1) for i in range(len(pol_hidden))
        ],
        nonlin=torch.nn.ReLU,
        weights_initializer=torch.nn.init.xavier_normal_,
        biases_initializer=None,
        output_nonlin=torch.nn.Tanh)

    pol = models.Policy(pol_model, maxU).float()

    # initalize experience dataset
    exp = utils.ExperienceDataset()

    # initialize dynamics optimizer
    opt1 = torch.optim.Adam(dyn.parameters(), 1e-3)

    # initialize policy optimizer
    opt2 = torch.optim.Adam(pol.parameters(), 1e-3)

    if use_cuda and torch.cuda.is_available():
        dyn = dyn.cuda()
        pol = pol.cuda()

    writer = tensorboardX.SummaryWriter()

    # callbacks
    def on_close():
        writer.close()

    atexit.register(on_close)

    # policy learning loop
    for it in range(100 + n_rnd):
        if it < n_rnd:
            pol_ = lambda x, t: maxU * (2 * np.random.rand(U, ) - 1)
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

        def on_iteration(i, loss, states, actions, rewards, opt, policy,
                         dynamics):
            writer.add_scalar('mc_pilco/episode_%d/training loss' % ps_it,
                              loss, i)
            if i % 100 == 0:
                states = states.transpose(0, 1).cpu().detach().numpy()
                actions = actions.transpose(0, 1).cpu().detach().numpy()
                rewards = rewards.transpose(0, 1).cpu().detach().numpy()
                utils.plot_trajectories(
                    states, actions, rewards, plot_samples=False)

        # train dynamics
        X, Y = exp.get_dynmodel_dataset(deltas=True, return_costs=learn_reward)
        dyn.set_dataset(
            torch.tensor(X).to(dyn.X.device).float(),
            torch.tensor(Y).to(dyn.X.device).float())
        train_regressor(
            dyn,
            2000,
            N_particles,
            True,
            opt1,
            log_likelihood=log_likelihood_loss)

        # sample initial states for policy optimization
        x0 = torch.tensor(exp.sample_states(N_particles, timestep=0)).to(
            dyn.X.device).float()
        x0 = x0 + 1e-1 * x0.std(0) * torch.randn_like(x0)
        x0 = x0.detach()
        utils.plot_rollout(x0, dyn, pol, H)

        # train policy
        print "Policy search iteration %d" % (ps_it + 1)
        algorithms.mc_pilco(
            x0,
            dyn,
            pol,
            H,
            opt2,
            exp,
            1000,
            pegasus=True,
            mm_states=True,
            mm_rewards=False,
            maximize=True,
            clip_grad=1.0)
        utils.plot_rollout(x0, dyn, pol, H)
        writer.add_scalar('robot/evaluation_loss',
                          torch.tensor(ret[2]).sum(), ps_it + 1)
