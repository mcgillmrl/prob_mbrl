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
"""Base dynamics model."""

import gym
import numpy as np
import torch

from enum import IntEnum
from gym.utils import seeding
from scipy.integrate import ode

from ..utils.classproperty import classproperty
from ..utils.angles import to_complex


class Integrator(IntEnum):
    FW_EULER = 0
    MIDPOINT = 1
    RUNGE_KUTTA = 2
    DOPRI5 = 3


class GymEnv(gym.Env):
    """Open AI gym  environment."""

    def __init__(self,
                 model,
                 reward_func=None,
                 measurement_noise=None,
                 angle_dims=[]):
        self.model = model

        self.seed()
        self.viewer = None
        self.state = None
        self.reward_func = reward_func
        self.steps = 0
        self.measurement_noise = None
        if measurement_noise is not None:
            self.measurement_noise = measurement_noise
        self.angle_dims = angle_dims

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, grads=False, **kwargs):
        x = torch.tensor(self.state)
        u = torch.tensor(action)
        if grads:
            x_next = self.model(x, u, 0, **kwargs)
        else:
            with torch.no_grad():
                x_next = self.model(x, u, 0, **kwargs)

        self.state = x_next.detach().cpu().numpy()

        if callable(self.reward_func):
            if grads:
                reward = self.reward_func(x_next, u)
            else:
                with torch.no_grad():
                    reward = self.reward_func(x_next, u).detach().cpu().numpy()
        else:
            reward = 0
        self.steps += 1
        done = False

        # post process state
        state = x_next
        if self.measurement_noise is not None:
            self.measurement_noise = self.measurement_noise.to(
                x_next.device, x_next.dtype)
            noise = self.measurement_noise * torch.randn_like(x_next)
            state = state + noise

        if self.angle_dims is not None:
            state = to_complex(state, self.angle_dims)
        state = state.detach().cpu().numpy()
        return state, reward, done, {}

    def reset(self, init_state, init_state_std):
        self.state = init_state + init_state_std * np.random.randn(
            *init_state.shape)
        state = self.state
        if self.angle_dims is not None:
            state = to_complex(torch.tensor(state), self.angle_dims).numpy()
        self.model.reset()
        return state


class DynamicsModel(torch.nn.Module):
    """Base dynamics model."""

    def __init__(self):
        super(DynamicsModel, self).__init__()
        self.solver = None

    def reset_parameters(self, initializer=torch.nn.init.normal_):
        """Resets all parameters that require gradients with random values.

        Args:
            initializer (callable): In-place function to initialize module
                parameters.

        Returns:
            self.
        """
        for p in self.parameters():
            if p.requires_grad:
                initializer(p)
        return self

    def reset(self):
        """ Resets the internal state of the solver """
        self.solver = None

    @classproperty
    def action_size(cls):
        """Action size (int)."""
        raise NotImplementedError

    @classproperty
    def state_size(cls):
        """State size (int)."""
        raise NotImplementedError

    def fit(self, X, U, dX, quiet=False, **kwargs):
        """Fits the dynamics model.

        Args:
            X (Tensor<N, state_size>): State trajectory.
            U (Tensor<N, action_size>): Action trajectory.
            dX (Tensor<N, state_size>): Next state trajectory.
            quiet (bool): Whether to print anything to screen or not.
        """
        raise NotImplementedError

    def dynamics(self, state, action, i, **kwargs):
        """Dynamics model function.

        Args:
            state (Tensor<..., state_size>): State distribution.
            u (Tensor<..., action_size>): Action vector(s).
            i (Tensor<...>): Time index.

        Returns:
            derivatives of current state wrt to time (Tensor<..., state_size>).
        """
        raise NotImplementedError

    def forward(self, state, action, i, int_method=Integrator.DOPRI5,
                **kwargs):
        """Dynamics model function.

        Args:
            state (Tensor<..., state_size>): State distribution.
            u (Tensor<..., action_size>): Action vector(s).
            i (int): Time index.
            encoding (int): StateEncoding enum.

        Returns:
            Next state distribution (Tensor<..., state_size>).
        """
        # we do numerical integration in doule precision
        if int_method == Integrator.FW_EULER:
            dmean = self.dynamics(state, action, i)
            next_state = state + dmean * self.dt
        elif int_method == Integrator.MIDPOINT:
            dmean = self.dynamics(state, action, i)
            mid = state + dmean * self.dt / 2
            dmid = self.dynamics(mid, action, i)
            next_state = state + dmid * self.dt
        elif int_method == Integrator.RUNGE_KUTTA:
            d1 = self.dynamics(state, action, i)
            d2 = self.dynamics(state + d1 * self.dt / 2, action, i)
            d3 = self.dynamics(state + d2 * self.dt / 2, action, i)
            d4 = self.dynamics(state + d3 * self.dt, action, i)
            next_state = state + (d1 + 2 * d2 + 2 * d3 + d4) * (self.dt / 6)
        elif int_method == Integrator.DOPRI5:
            # note that this is not currently differentiable
            def dyn_fn(t, z_t):
                z_t = torch.tensor(z_t).to(state.dtype)
                return self.dynamics(z_t, action, t).detach().numpy()

            if self.solver is None:
                self.solver = ode(dyn_fn).set_integrator(
                    'dopri5', atol=1e-9, rtol=1e-9)
            solver = self.solver.set_initial_value(state.detach().numpy())
            t = solver.t
            t_end = t + self.dt
            while solver.successful and solver.t < t_end:
                solver.integrate(solver.t + self.dt)
            next_state = torch.tensor(solver.y).to(state.dtype)

        return next_state
