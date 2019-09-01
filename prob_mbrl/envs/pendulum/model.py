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
"""Pendulum dynamics model."""

import numpy as np
import torch
from torch.nn import Parameter

from ..base import DynamicsModel
from ...utils.classproperty import classproperty


class PendulumModel(DynamicsModel):
    """Pendulum dynamics model.

    Note:
        state: [theta, theta']
        action: [torque]
        theta: 0 is pointing up and increasing counter-clockwise.
    """

    def __init__(self, dt=0.1, m=1.0, l=1.0, mu=0.01, g=9.82):  # noqa: E741
        """Constructs PendulumDynamicsModel.

        Args:
            dt (float): Time step [s].
            m (float): Pendulum mass [kg].
            l (float): Pendulum length [m].
            g (float): Gravity acceleration [m/s^2].
        """
        super(PendulumModel, self).__init__()
        self.dt = Parameter(torch.tensor(dt), requires_grad=False)
        self.m = Parameter(torch.tensor(m), requires_grad=False)
        self.l = Parameter(torch.tensor(l), requires_grad=False)  # noqa: E741
        self.mu = Parameter(torch.tensor(mu), requires_grad=False)
        self.g = Parameter(torch.tensor(g), requires_grad=False)

    @classproperty
    def action_size(cls):
        """Action size (int)."""
        return 1

    @classproperty
    def state_size(cls):
        """State size (int)."""
        return 2

    @classproperty
    def angular_indices(cls):
        """Column indices of angular states (Tensor)."""
        return torch.tensor([0]).long()

    @classproperty
    def non_angular_indices(cls):
        """Column indices of non-angular states (Tensor)."""
        return torch.tensor([1]).long()

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

    def dynamics(self, z, u, i):
        """Dynamics model function.

        Args:
            z (Tensor<..., state_size>): State distribution.
            u (Tensor<..., action_size>): Action vector(s).
            i (Tensor<...>): Time index.

        Returns:
            derivatives of current state wrt to time (Tensor<..., state_size>).
        """

        if not torch.is_grad_enabled():
            if isinstance(z, torch.Tensor):
                z = z.detach().numpy()
            if isinstance(u, torch.Tensor):
                u = u.detach().numpy()
            m = self.m.numpy()
            l = self.l.numpy()  # noqa: E741
            mu = self.mu.numpy()
            g = self.g.numpy()
        else:
            m = self.m
            l = self.l  # noqa: E741
            mu = self.mu
            g = self.g

        theta = z[..., 0]
        theta_dot = z[..., 1]
        torque = u[..., 0]

        sin_theta = theta.sin() if torch.is_grad_enabled() else np.sin(theta)

        # Define acceleration.
        temp = m * l
        theta_dot_dot = torque - mu * theta_dot - 0.5 * temp * g * sin_theta
        theta_dot_dot = 3 * theta_dot_dot / (temp * l)

        if not torch.is_grad_enabled():
            dz = np.zeros_like(z)
            dz[..., 0] = theta_dot
            dz[..., 1] = theta_dot_dot
            return dz

        return torch.stack([
            theta_dot,
            theta_dot_dot,
        ], dim=-1)
