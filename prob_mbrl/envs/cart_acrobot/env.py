# Copyright (C) 2018, Juan Camilo Gamboa Higuera
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
"""Double cartpole environment."""

import torch
import numpy as np

from gym import spaces

from .model import CartAcrobotModel
from ..base import GymEnv
from ...utils import angles


class CartAcrobotReward(torch.nn.Module):
    def __init__(self,
                 pole1_length=0.6,
                 pole2_length=0.6,
                 target=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                 Q=8.0 * torch.eye(2),
                 R=1e-4 * torch.eye(1)):
        super(CartAcrobotReward, self).__init__()
        self.Q = torch.nn.Parameter(torch.tensor(Q), requires_grad=False)
        self.R = torch.nn.Parameter(torch.tensor(R), requires_grad=False)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        self.target = torch.nn.Parameter(
            torch.tensor(target), requires_grad=False)
        self.pole1_length = torch.nn.Parameter(
            torch.tensor(pole1_length), requires_grad=False)
        self.pole2_length = torch.nn.Parameter(
            torch.tensor(pole2_length), requires_grad=False)

    def forward(self, x, u):
        x = x.to(device=self.Q.device, dtype=self.Q.dtype)
        u = u.to(device=self.Q.device, dtype=self.Q.dtype)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if u.dim() == 1:
            u = u.unsqueeze(0)
        # compute the distance between the tip of the pole and the target tip
        # location
        targeta = angles.to_complex(self.target, [2, 4])
        target_tip_xy = torch.cat([
            targeta[:, 0:1] - self.pole1_length * targeta[:, 4:5] -
            self.pole2_length * targeta[:, 5:6],
            self.pole1_length * targeta[:, 6:7] +
            self.pole2_length * targeta[:, 7:8]
        ],
                                  dim=-1)
        xa = angles.to_complex(x, [2, 4])
        pole_tip_xy = torch.cat([
            xa[:, 0:1] - self.pole1_length * xa[:, 4:5] -
            self.pole2_length * xa[:, 5:6],
            self.pole1_length * xa[:, 6:7] + self.pole2_length * xa[:, 7:8]
        ],
                                dim=-1)

        pole_tip_xy = pole_tip_xy.unsqueeze(
            0) if pole_tip_xy.dim() == 1 else pole_tip_xy
        target_tip_xy = target_tip_xy.unsqueeze(
            0) if target_tip_xy.dim() == 1 else target_tip_xy

        delta = pole_tip_xy - target_tip_xy
        delta = delta / (2 * (self.pole1_length + self.pole2_length))
        cost = 0.5 * ((delta.mm(self.Q) * delta).sum(-1, keepdim=True) +
                      (u.mm(self.R) * u).sum(-1, keepdim=True))
        # reward is negative cost.
        # optimizing the exponential of the negative cost is equivalent to
        # doing inference to maximize rewards (high reward trajectories
        # should be more likely), assuming conditionally independent rewards
        reward = (-cost).exp()
        return reward


class CartAcrobot(GymEnv):
    """Open AI gym double cartpole environment."""

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 50,
    }

    def __init__(self, model=None, reward_func=None):
        if model is None:
            model = CartAcrobotModel()
        # init parent class
        reward_func = reward_func if callable(
            reward_func) else CartAcrobotReward(
                pole1_length=model.l1, pole2_length=model.l2)
        measurement_noise = torch.tensor([0.01] * 6)
        super(CartAcrobot, self).__init__(model, reward_func,
                                          measurement_noise)

        # init this class
        high = np.array([5])
        self.action_space = spaces.Box(-high, high, dtype=np.float32)

        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            2 * np.pi,
            np.finfo(np.float32).max,
            2 * np.pi,
            np.finfo(np.float32).max,
        ])
        if self.angle_dims is not None:
            low = angles.to_complex(torch.tensor(-high),
                                    self.angle_dims).numpy()
            high = angles.to_complex(torch.tensor(high),
                                     self.angle_dims).numpy()
        else:
            low = -high
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32)

    def reset(self,
              init_state=np.array([0, 0, np.pi, 0, np.pi, 0]),
              init_state_std=2e-1):
        return super(CartAcrobot, self).reset(init_state, init_state_std)

    def render(self, mode="human"):
        screen_width = 1000
        screen_height = 600

        world_width = 5.0
        scale = screen_width / world_width
        carty = 200  # TOP OF CART
        polewidth1 = 10.0 * (self.model.mp1 / 0.5)
        polewidth2 = 10.0 * (self.model.mp2 / 0.5)
        polelen1 = scale * self.model.l1
        polelen2 = scale * self.model.l2
        cartwidth = 50.0 * torch.sqrt(self.model.mc / 0.5)
        cartheight = 30.0 * torch.sqrt(self.model.mc / 0.5)

        if self.state is None:
            return None

        x, _, theta1, _, theta2, _ = self.state
        cartx = x * scale + screen_width / 2.0  # MIDDLE OF CART

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.window.set_vsync(False)

            l, r, t, b = (-cartwidth / 2, cartwidth / 2, cartheight / 2,
                          -cartheight / 2)
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            l, r, t, b = (-polewidth1 / 2, polewidth1 / 2,
                          polelen1 - polewidth1 / 2, -polewidth1 / 2)
            pole1 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole1.set_color(0.8, 0.6, 0.4)
            self.pole1trans = rendering.Transform(translation=(0, axleoffset))
            pole1.add_attr(self.pole1trans)
            pole1.add_attr(self.carttrans)
            self.viewer.add_geom(pole1)

            self.axle1 = rendering.make_circle(polewidth1 / 2)
            self.axle1.add_attr(self.pole1trans)
            self.axle1.add_attr(self.carttrans)
            self.axle1.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle1)

            l, r, t, b = (-polewidth2 / 2, polewidth2 / 2,
                          polelen2 - polewidth2 / 2, -polewidth2 / 2)
            pole2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole2.set_color(0.8, 0.6, 0.4)
            self.pole2trans = rendering.Transform(
                translation=(0, polelen1 - axleoffset))
            pole2.add_attr(self.pole2trans)
            pole2.add_attr(self.pole1trans)
            pole2.add_attr(self.carttrans)
            self.viewer.add_geom(pole2)

            self.axle2 = rendering.make_circle(polewidth2 / 2)
            self.axle2.add_attr(self.pole2trans)
            self.axle2.add_attr(self.pole1trans)
            self.axle2.add_attr(self.carttrans)
            self.axle2.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle2)

            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        self.carttrans.set_translation(cartx, carty)
        self.pole1trans.set_rotation(theta1)
        self.pole2trans.set_rotation(theta2 - theta1)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
