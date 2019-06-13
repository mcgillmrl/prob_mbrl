import numpy as np
import os
import torch

from xml.etree import ElementTree as ET
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco import InvertedPendulumEnv as BaseInvertedPendulumEnv


class Cartpole(BaseInvertedPendulumEnv):
    def __init__(self,
                 cart_mass=0.5,
                 pole_mass=0.5,
                 pole_length=1.0,
                 cart_damping=0.1,
                 max_control=10.0,
                 gravity=9.82,
                 dt=0.1,
                 frame_skip=1):
        base_dir = os.path.dirname(__file__)
        path = os.path.join(base_dir, "assets", 'cartpole.xml')
        xmldoc = ET.parse(path)
        root = xmldoc.getroot()
        nsubsteps = dt / 0.01
        for node in root.iter('option'):
            node.set('gravity', '{0} {1} {2}'.format(0.0, 0.0, -gravity))
            node.set('timestep', str(dt / nsubsteps))

        for node in root.iter('geom'):
            name = node.get('name')
            if name == 'pole':
                node.set('mass', str(pole_mass))
                node.set('fromto',
                         '0.0 0.0 0.0 0.0 0.0 {0}'.format(pole_length))

                r = float(node.get('size'))
                new_r = np.sqrt((2 * pole_mass * r**2) / pole_length)
                new_r = max(min(0.5 * pole_length, new_r), 0.01)
                node.set('size', str(new_r))
            elif name == 'cart':
                node.set('mass', str(cart_mass))
                dims = np.array(
                    [float(x) for x in node.get('size').split(' ')])
                new_dims = dims * ((2 * cart_mass)**(1.0 / 3.0))
                node.set('size', '{0} {1} {2}'.format(*new_dims))

        for node in root.iter('joint'):
            name = node.get('name')
            if name == 'rail':
                node.set('damping', cart_damping)

        for node in root.iter('motor'):
            name = node.get('name')
            if name == 'slide':
                node.set('gear', '{0}'.format(str(max_control)))
                node.set('ctrlrange', '-1 1')

        tmppath = '/tmp/cartpole_{0}_{1}_{2}_{3}_{4}_{5}.xml'.format(
            cart_mass, pole_mass, pole_length,
            cart_damping, max_control, gravity, dt)
        xmldoc.write(tmppath)
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, tmppath, frame_skip)
        self.sim.nsubsteps = nsubsteps

    def _reward(self, state, action):
        # reward for maintaining the pole upright
        angle_rw = (np.cos(state[1]) + 1) / 2.0

        # reward for maintaining the cart centered
        scale = np.sqrt(-2 * np.log(0.1)) / 2.0
        cart_rw = (1 + np.exp(-0.5 * (state[0] * scale)**2)) / 2

        # control penalty (reduces reward for larger actions)
        control_rw = (4 + np.maximum(np.zeros_like(action), 1 - action**2)) / 5

        # velocity penalty (halves the reward if spinning too fast)
        vel_rw = (1 + np.exp(-0.5 * state[3]**2)) / 2

        return angle_rw * cart_rw * control_rw * vel_rw

    def reward_func(self, state, action):
        # reward for maintaining the pole upright
        angle_rw = (torch.cos(state[..., 1:2]) + 1) / 2.0

        # reward for maintaining the cart centered
        scale = np.sqrt(-2 * np.log(0.1)) / 2.0
        cart_rw = (1 + torch.exp(-0.5 * (state[..., 0:1] * scale)**2)) / 2

        # control penalty (reduces reward for larger actions)
        control_rw = (4 + torch.max(torch.zeros_like(action), 1 - action**2)) / 5

        # velocity penalty (halves the reward if spinning too fast)
        vel_rw = (1 + torch.exp(-0.5 * state[..., 3:4]**2)) / 2

        return angle_rw * cart_rw * control_rw * vel_rw

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        reward = self._reward(ob, action)

        # if spinning too much or numerical integration errors
        done = not np.isfinite(ob).all() or (np.abs(ob[1]) > 4 * np.pi)

        return ob, reward, done, {}

    def reset_model(self, init_state=[0, np.pi, 0, 0], init_state_std=1e-20):
        sigma = np.ones_like(init_state) * init_state_std
        qpos = self.init_qpos + self.np_random.normal(
            size=self.model.nq) * sigma[:2]
        qvel = self.init_qvel + self.np_random.normal(
            size=self.model.nv) * sigma[:2]
        self.set_state(qpos + init_state[:2], qvel + init_state[2:])
        return self._get_obs()