import numpy as np
import torch

from functools import partial

from kusanagi.shell import cartpole
from kusanagi.base import ExperienceDataset, apply_controller
from kusanagi.ghost.control import RandPolicy

from prob_mbrl import utils, models, algorithms, losses, train_regressor
torch.set_num_threads(2)


def forward(states, actions, dynamics, **kwargs):
    deltas, rewards = dynamics(
        (states, actions), return_samples=True,
        separate_outputs=True, **kwargs)
    next_states = states + deltas
    return next_states, rewards


def reward_fn(states, target, Q, angle_dims):
    states = utils.to_complex(states, angle_dims)
    reward = losses.quadratic_saturating_loss(states, target, Q)
    return reward


# parameters
H = 25
N_particles = 50
dyn_components = 2
dyn_hidden = [200]*2
pol_hidden = [200]*2

# initialize environment
env = cartpole.Cartpole()

# initialize reward/cost function
target = torch.tensor([0, 0, 0, np.pi]).float()
D = target.shape[-1]
U = 1
learn_reward = False
maxU = np.array([10.0])
angle_dims = torch.tensor([3]).long()
target = utils.to_complex(target, angle_dims)
Da = target.shape[-1]
Q = torch.zeros(Da, Da).float()
Q[0, 0] = 1
Q[0, -2] = env.l
Q[-2, 0] = env.l
Q[-2, -2] = env.l**2
Q[-1, -1] = env.l**2
Q /= 0.1
if learn_reward:
    reward_func = None
else:
    reward_func = partial(
        reward_fn, target=target, Q=Q, angle_dims=angle_dims)


# initialize dynamics model
dynE = 2*(D+1) if learn_reward else 2*D
dyn_model = models.dropout_mlp(
            Da+U, (dynE+1)*dyn_components, dyn_hidden,
            dropout_layers=[models.modules.CDropout(0.1, 0.1)
                            for i in range(len(dyn_hidden))],
            nonlin=torch.nn.ReLU,
            weights_initializer=torch.nn.init.xavier_normal_,
            biases_initializer=partial(torch.nn.init.uniform_, a=-1.0, b=1.0),
        )
dyn = models.DynamicsModel(
    dyn_model, reward_func=reward_func,
    angle_dims=angle_dims,
    output_density=models.MixtureDensity(dynE/2, dyn_components)).float()

# initalize policy
pol_model = models.dropout_mlp(
        Da, U, pol_hidden,
        dropout_layers=[models.modules.BDropout(0.1)
                        for i in range(len(pol_hidden))],
        nonlin=torch.nn.ReLU,
        output_nonlin=torch.nn.Tanh)

pol = models.Policy(pol_model, maxU, angle_dims=angle_dims).float()
randpol = RandPolicy(maxU)

# initalize experience dataset
exp = ExperienceDataset()

# initialize policy optimizer
params = filter(lambda p: p.requires_grad, pol.parameters())
opt = torch.optim.Adam(params, 1e-4, amsgrad=True)

# define functions required for rollouts
forward_fn = partial(forward, dynamics=dyn)

# collect initial random experience
for rand_it in range(1):
    ret = apply_controller(
        env, randpol, H,
        callback=lambda *args, **kwargs: env.render())
    exp.append_episode(*ret)

# policy learning loop
for ps_it in range(100):
    # apply policy
    ret = apply_controller(
        env, pol, H,
        callback=lambda *args, **kwargs: env.render())
    exp.append_episode(*ret)

    # train dynamics
    X, Y = exp.get_dynmodel_dataset(deltas=True, return_costs=learn_reward)
    dyn.set_dataset(
        torch.tensor(X).to(dyn.X.device).float(),
        torch.tensor(Y).to(dyn.X.device).float())
    train_regressor(
        dyn, 1000, N_particles, True,
        log_likelihood=losses.gaussian_mixture_log_likelihood)

    # sample initial states for policy optimization
    x0 = torch.tensor(
        exp.sample_states(N_particles, timestep=0)).to(dyn.X.device).float()
    x0 += 1e-2*x0.std(0)*torch.randn_like(x0)
    utils.plot_rollout(x0, forward_fn, pol, H)

    # train policy
    print "Policy search iteration %d" % (ps_it+1)
    algorithms.mc_pilco(
        x0, forward_fn, dyn, pol, H, opt, exp=exp,
        maximize=False, pegasus=True, mm_states=False,
        mm_rewards=False, mpc=False, max_steps=25)
    utils.plot_rollout(x0, forward_fn, pol, H)
