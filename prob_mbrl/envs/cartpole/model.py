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
"""Cartpole dynamics model."""

import torch
from torch.nn import Parameter

from ..base import DynamicsModel
from ...utils.classproperty import classproperty


class CartpoleModel(DynamicsModel):
    """Cartpole dynamics model.

    Note:
        state: [x, x', theta, theta']
        action: [F]
        theta: 0 is pointing up and increasing clockwise.
    """

    def __init__(self, dt=0.1, mc=0.5, mp=0.5, lp=0.5, mu=0.1, g=9.82):
        """Constructs a CartpoleModel.

        Args:
            dt (float): Time step [s].
            mc (float): Cart mass [kg].
            mp (float): Pendulum mass [kg].
            l (float): Pendulum length [m].
            mu (float): Coefficient of friction [dimensionless].
            g (float): Gravity acceleration [m/s^2].
        """
        super(CartpoleModel, self).__init__()
        self.dt = Parameter(torch.tensor(dt), requires_grad=False)
        self.mc = Parameter(torch.tensor(mc), requires_grad=False)
        self.mp = Parameter(torch.tensor(mp), requires_grad=False)
        self.lp = Parameter(torch.tensor(lp), requires_grad=False)
        self.mu = Parameter(torch.tensor(mu), requires_grad=False)
        self.g = Parameter(torch.tensor(g), requires_grad=False)

    @classproperty
    def action_size(cls):
        """Action size (int)."""
        return 1

    @classproperty
    def state_size(cls):
        """State size (int)."""
        return 4

    @classproperty
    def angular_indices(cls):
        """Column indices of angular states (Tensor)."""
        return torch.tensor([2]).long()

    @classproperty
    def non_angular_indices(cls):
        """Column indices of non-angular states (Tensor)."""
        return torch.tensor([0, 1, 3]).long()

    def fit(self, X, U, dX, quiet=False, **kwargs):
        """Fits the dynamics model.

        Args:
            X (Tensor<N, state_size>): State trajectory.
            U (Tensor<N, action_size>): Action trajectory.
            dX (Tensor<N, state_size>): Next state trajectory.
            quiet (bool): Whether to print anything to screen or not.
        """
        # No need: this is an exact dynamics model.
        pass

    def dynamics(self, z, u, i, **kwargs):
        """Dynamics model function.

        Args:
            z (Tensor<..., state_size>): State distribution.
            u (Tensor<..., action_size>): Action vector(s).
            i (Tensor<...>): Time index.

        Returns:
            derivatives of current state wrt to time (Tensor<..., state_size>).
        """
        mc = self.mc
        mp = self.mp
        lp = self.lp
        mu = self.mu
        g = self.g

        x_dot = z[..., 1]
        theta = z[..., 2]
        theta_dot = z[..., 3]
        F = u[..., 0]

        sin_theta = theta.sin()
        cos_theta = theta.cos()

        a0 = mp * lp * theta_dot**2 * sin_theta
        a1 = g * sin_theta
        a2 = F - mu * x_dot
        a3 = 4 * (mc + mp) - 3 * mp * cos_theta**2

        theta_dot_dot = -3 * (a0 * cos_theta + 2 * (
            (mc + mp) * a1 + a2 * cos_theta)) / (lp * a3)
        x_dot_dot = (2 * a0 + 3 * mp * a1 * cos_theta + 4 * a2) / a3

        return torch.stack([x_dot, x_dot_dot, theta_dot, theta_dot_dot],
                           dim=-1)
