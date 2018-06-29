import numpy as np
import sys
import torch
torch.set_num_threads(2)

from functools import partial

from kusanagi.shell import cartpole
from kusanagi.base import ExperienceDataset, apply_controller
from kusanagi.ghost.control import RandPolicy

sys.path.append('..')
from prob_mbrl import algorithms, models, train_regressor, losses, utils


def forward(states, actions, dynamics, measurement_noise=True,
            z_states=None, z_rewards=None,**kwargs):
    
    (deltas, deltas_std), (rewards, rewards_std) = dyn(
        (states, actions), separate_outputs=True, **kwargs)
    next_states = states + deltas
    if measurement_noise:
        z1 = z_states if z_states is not None else torch.randn_like(next_states)
        next_states += z1*deltas_std
        #z2 = z_rewards if z_rewards is not None else torch.randn(next_states.shape[0], 1)
        #rewards += z2*rewards_std
    return next_states, rewards


def reward_func(states, target, Q, angle_dims):
    states = utils.to_complex(states, angle_dims)
    reward = losses.quadratic_saturating_loss(states, target, Q)
    return reward, torch.zeros_like(reward)


# experiment parameters
n_rnd = 1
n_opt = 25
H = 25
N_particles = 100
learn_reward=True
angle_dims = torch.tensor([3]).long()
target = torch.tensor([0,0,0,np.pi]).float()
target = utils.to_complex(target, angle_dims)
Da = target.shape[-1]
Q = torch.zeros(Da, Da).float()
Q[0, 0] = 1
Q[0, -2] = env.l
Q[-2, 0] = env.l
Q[-2, -2] = env.l**2
Q[-1, -1] = env.l**2
Q /= 0.1
maxU = torch.tensor([10.0])
D = target.shape[-1]
U = maxU.shape[-1]

# init environment
env = cartpole.Cartpole()

# init reward/cost function
if learn_reward:
    dynE = 2*(D+1)
    reward_func = None
else:
    dynE = 2*D
    reward_func = partial(reward_func, target=target, Q=Q, angle_dims=angle_dims)

# init dynamics model (heteroscedastic noise)
dyn = models.DynamicsModel(
    models.dropout_mlp(
        D+U, dynE, [200]*2,
        dropout_layers=[models.modules.CDropout(0.1)]*2,
        nonlin=torch.nn.ReLU), reward_func=reward_func).float()
forward_fn = partial(forward, dynamics=dyn)

# init policy
pol = models.Policy(
    models.dropout_mlp(
        D, U, output_nonlin=torch.nn.Tanh,
        dropout_layers=[models.modules.BDropout(0.1)]*2), maxU).float()
randpol = RandPolicy(maxU)

# init experience dataset
exp = ExperienceDataset()

# init policy optimizer
params = filter(lambda p: p.requires_grad, pol.parameters())
opt = torch.optim.Adam(params, 1e-3, amsgrad=True)

def cb(*args, **kwargs):
    env.render()

for rand_it in range(n_rnd):
    ret = apply_controller(
        env, randpol, H, callback=lambda *args, **kwargs: env.render())
    exp.append_episode(*ret)

for ps_it in range(n_opt):
    # apply policy
    ret = apply_controller(
        env, pol, H, callback=lambda *args, **kwargs: env.render())
    exp.append_episode(*ret)

    # train dynamics
    X, Y = exp.get_dynmodel_dataset(deltas=True, return_costs=learn_reward)
    dyn.set_dataset(torch.tensor(X).to(dyn.X.device).float(), torch.tensor(Y).to(dyn.X.device).float())  
    train_regressor(dyn, 1000, N_particles, True)
    x0 = torch.tensor(exp.sample_initial_state(N_particles)).to(dyn.X.device).float()
    x0 += 1e-2*x0.std(0)*torch.randn_like(x0)
    utils.plot_rollout(x0, forward_fn, pol, H)
    
    # train policy
    print "Policy search iteration %d" % (ps_it+1)
    algorithms.mc_pilco(x0, forward_fn, dyn, pol, H, opt, exp=exp,
             maximize=False, pegasus=True, mm_states=True,
             mm_rewards=True, angle_dims=angle_dims)
    utils.plot_rollout(x0, forward_fn, pol, H)
    