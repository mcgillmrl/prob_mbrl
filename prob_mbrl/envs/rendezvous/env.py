# Copyright (C) 2018, Anass Al, Juan Camilo Gamboa Higuera
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
"""Multi-vehicle rendezvous environment."""

import torch
import numpy as np

from gym import spaces

from .model import RendezvousModel
from ..base import GymEnv


class RendezvousReward(torch.nn.Module):
    def __init__(self, Q=1.0 * torch.eye(4), R=1.0 * torch.eye(4)):
        super(RendezvousReward, self).__init__()
        self.Q = torch.nn.Parameter(torch.tensor(Q), requires_grad=False)
        self.R = torch.nn.Parameter(torch.tensor(R), requires_grad=False)

    def forward(self, x, u):
        x = x.to(device=self.Q.device, dtype=self.Q.dtype)
        u = u.to(device=self.Q.device, dtype=self.Q.dtype)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if u.dim() == 1:
            u = u.unsqueeze(0)
        state_1 = torch.cat([x[:, :2], x[:, 4:6]], -1)
        state_2 = torch.cat([x[:, 2:4], x[:, 6:8]], -1)
        delta = state_1 - state_2
        cost = (delta.matmul(self.Q) * delta).sum(-1) + (
            u.matmul(self.R) * u).sum(-1)
        reward = -cost
        return reward


class Rendezvous(GymEnv):
    """Open AI gym multi-vehicle rendezvous environment."""

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 30,
    }

    def __init__(self, model=RendezvousModel(), reward_func=None):
        # init parent class
        reward_func = reward_func if callable(
            reward_func) else RendezvousReward()
        super(Rendezvous, self).__init__(model, reward_func)

        # init this class
        high = np.array([100] * 4)
        self.action_space = spaces.Box(low=-high, high=high)

        high = np.array([np.finfo(np.float32).max] * 8)
        self.observation_space = spaces.Box(low=-high, high=high)

    def reset(self,
              init_state=np.array(
                  [-10.0, -10.0, 10.0, 10.0, 0.0, -0.0, 0.0, 0.0])):
        self.state = init_state
        self.state += 1e-2 * np.random.randn(*self.state.shape)
        self.steps = 0
        return self.state

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-15.0, 15.0, -15.0, 15.0)

            vehicle_0 = rendering.make_circle(0.5)
            vehicle_0.set_color(1.0, 0.0, 0.0)
            self.vehicle_0_transform = rendering.Transform()
            vehicle_0.add_attr(self.vehicle_0_transform)
            self.viewer.add_geom(vehicle_0)

            vehicle_1 = rendering.make_circle(0.5)
            vehicle_1.set_color(0.0, 0.0, 1.0)
            self.vehicle_1_transform = rendering.Transform()
            vehicle_1.add_attr(self.vehicle_1_transform)
            self.viewer.add_geom(vehicle_1)

        x_0, y_0, x_1, y_1, _, _, _, _ = self.state
        self.vehicle_0_transform.set_translation(x_0, y_0)
        self.vehicle_1_transform.set_translation(x_1, y_1)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
