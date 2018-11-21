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
import torch
from torch.nn import Parameter

from ..base import DynamicsModel
from ...utils.classproperty import classproperty


class RendezvousModel(DynamicsModel):
    """Multi-vehicle rendezvous dynamics model.

    Note:
        state: [x_0, y_0, x_1, y_1, x_0_dot, y_0_dot, x_1_dot, y_1_dot]
        action: [F_x_0, F_y_0, F_x_1, F_y_1]
    """

    def __init__(self, dt=0.1, m=1.0, alpha=0.1):
        """Constructs RendezvousDynamicsModel.

        Args:
            dt (float): Time step [s].
            m: Vehicle mass [kg].
            alpha: Friction coefficient.
        """
        super(RendezvousModel, self).__init__()
        self.dt = Parameter(torch.tensor(dt), requires_grad=False)
        self.m = Parameter(torch.tensor(m), requires_grad=True)
        self.alpha = Parameter(torch.tensor(alpha), requires_grad=True)

    @classproperty
    def action_size(cls):
        """Action size (int)."""
        return 4

    @classproperty
    def state_size(cls):
        """State size (int)."""
        return 8

    @classproperty
    def angular_indices(cls):
        """Column indices of angular states (Tensor)."""
        return torch.tensor([]).long()

    @classproperty
    def non_angular_indices(cls):
        """Column indices of non-angular states (Tensor)."""
        return torch.arange(8).long()

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

        return torch.stack(
            [
                z[..., 4],
                z[..., 5],
                z[..., 6],
                z[..., 7],
                self._acceleration(z[..., 4], u[..., 0]),
                self._acceleration(z[..., 5], u[..., 1]),
                self._acceleration(z[..., 6], u[..., 2]),
                self._acceleration(z[..., 7], u[..., 3]),
            ],
            dim=-1)

    def _acceleration(self, x_dot, u):
        x_dot_dot = x_dot * (1 - self.alpha * self.dt / self.m)
        x_dot_dot += u * self.dt / self.m
        return x_dot_dot
