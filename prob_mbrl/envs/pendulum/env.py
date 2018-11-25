# Copyright (C) 2018, Anass Al
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
"""Pendulum environment."""

import torch
import numpy as np

from gym import spaces

from .model import PendulumModel
from ..base import GymEnv
from ...utils import angles


class PendulumReward(torch.nn.Module):
    def __init__(self,
                 pole_length=0.5,
                 target=torch.tensor([np.pi, 0]),
                 Q=4.0 * torch.eye(2),
                 R=1e-4 * torch.eye(1)):
        super(PendulumReward, self).__init__()
        self.Q = torch.nn.Parameter(torch.tensor(Q), requires_grad=False)
        self.R = torch.nn.Parameter(torch.tensor(R), requires_grad=False)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        self.target = torch.nn.Parameter(
            torch.tensor(target), requires_grad=False)
        self.pole_length = torch.nn.Parameter(
            torch.tensor(pole_length), requires_grad=False)

    def forward(self, x, u):
        x = x.to(device=self.Q.device, dtype=self.Q.dtype)
        u = u.to(device=self.Q.device, dtype=self.Q.dtype)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if u.dim() == 1:
            u = u.unsqueeze(0)
        # compute the distance between the tip of the pole and the target tip
        # location
        targeta = angles.to_complex(self.target, [0])
        target_tip_xy = torch.cat([
            self.pole_length * targeta[:, 1:2],
            -self.pole_length * targeta[:, 2:3]
        ],
                                  dim=-1)
        xa = angles.to_complex(x, [0])
        pole_tip_xy = torch.cat(
            [self.pole_length * xa[:, 1:2], -self.pole_length * xa[:, 2:3]],
            dim=-1)

        # normalized distance so that cost at [0 ,0] is 1
        delta = (pole_tip_xy - target_tip_xy)
        delta = delta / (2 * self.pole_length)

        # compute cost
        cost = 0.5 * ((delta.mm(self.Q) * delta).sum(-1, keepdim=True) +
                      (u.mm(self.R) * u).sum(-1, keepdim=True))

        # reward is negative cost.
        # optimizing the exponential of the negative cost is equivalent to
        # doing inference to maximize rewards (high reward trajectories
        # should be more likely), assuming conditionally independent rewards
        reward = (-cost).exp()
        return reward


class Pendulum(GymEnv):
    """Open AI gym pendulum environment.

    Based on the OpenAI gym Pendulum-v0 environment, but with more
    custom dynamics for a better ground truth.
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 30,
    }

    def __init__(self, model=PendulumModel(), reward_func=None):
        # init parent class
        reward_func = reward_func if callable(reward_func) else PendulumReward(
            pole_length=model.l)
        measurement_noise = torch.tensor([0.1, 0.01])
        super(Pendulum, self).__init__(model, reward_func, measurement_noise)

        # init this class
        high = np.array([2.5])
        self.action_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        high = np.array([np.pi, np.finfo(np.float32).max])
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float32)

    def reset(self, init_state=np.array([0.0, 0.0]), init_state_std=2e-1):
        self.state = init_state + init_state_std * np.random.randn(
            *init_state.shape)
        return self.state

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

            rod = rendering.make_capsule(1.0 * self.model.l,
                                         0.2 * torch.sqrt(self.model.m / 1.0))
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)

            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)

        theta, _ = self.state
        self.pole_transform.set_rotation(theta - np.pi / 2)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
