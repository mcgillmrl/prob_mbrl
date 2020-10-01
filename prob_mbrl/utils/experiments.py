import argparse
import datetime
import numpy as np
import os
import torch

from prob_mbrl import envs
from .core import load_csv


def get_argument_parser(title=""):
    parser = argparse.ArgumentParser(title)
    parser.add_argument('-e', '--env', type=str, default="Cartpole")
    parser.add_argument('-o',
                        '--output_folder',
                        type=str,
                        default="~/.prob_mbrl/")
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--n_initial_epi', type=int, default=0)
    parser.add_argument('--load_from', type=str, default=None)
    parser.add_argument('--pred_H', type=int, default=15)
    parser.add_argument('--control_H', type=int, default=40)
    parser.add_argument('--discount_factor', type=str, default=None)
    parser.add_argument('--prioritized_replay', action='store_true')
    parser.add_argument('--timesteps_to_sample', type=load_csv, default=0)
    parser.add_argument('--mm_groups', type=int, default=None)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--dyn_lr', type=float, default=1e-4)
    parser.add_argument('--dyn_opt_iters', type=int, default=2000)
    parser.add_argument('--dyn_batch_size', type=int, default=100)
    parser.add_argument('--dyn_drop_rate', type=float, default=0.1)
    parser.add_argument('--dyn_components', type=int, default=1)
    parser.add_argument('--dyn_shape', type=load_csv, default=[200, 200])

    parser.add_argument('--pol_lr', type=float, default=1e-3)
    parser.add_argument('--pol_clip', type=float, default=1.0)
    parser.add_argument('--pol_drop_rate', type=float, default=0.1)
    parser.add_argument('--pol_opt_iters', type=int, default=1000)
    parser.add_argument('--pol_batch_size', type=int, default=100)
    parser.add_argument('--ps_iters', type=int, default=100)
    parser.add_argument('--pol_shape', type=load_csv, default=[200, 200])

    parser.add_argument('--plot_level', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--learn_reward', action='store_true')
    parser.add_argument('--keep_best', action='store_true')
    parser.add_argument('--stop_when_done', action='store_true')
    parser.add_argument('--expl_noise', type=float, default=0.0)
    parser.add_argument('--resampling_period', type=int, default=499)

    return parser


def init_env(env_name, seed):
    # initialize environment
    torch.manual_seed(seed)
    np.random.seed(seed)
    if env_name in envs.__all__:
        env = envs.__dict__[env_name]()
    else:
        import gym
        env = gym.make(env_name)

    return env


def init_output_folder(env, output_folder):
    env_name = env.spec.id if env.spec is not None else env.__class__.__name__
    output_folder = os.path.expanduser(output_folder)

    results_folder = os.path.join(
        output_folder, "mc_pilco_mm", env_name,
        datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S.%f"))
    try:
        os.makedirs(results_folder)
    except OSError:
        pass

    return results_folder
