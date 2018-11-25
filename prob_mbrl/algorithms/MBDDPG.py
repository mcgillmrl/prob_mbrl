"""
    Model based DDPG based on the code available 
    here: https://github.com/sfujim/TD3
"""
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from prob_mbrl import models, losses, utils

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(models.Policy):
    def __init__(self, state_dim, action_dim, max_action,
                 pol_hidden=[200] * 2):
        pol_model = models.mlp(
            state_dim,
            action_dim,
            pol_hidden,
            dropout_layers=[
                models.modules.BDropout(0.1) for i in range(len(pol_hidden))
            ],
            nonlin=torch.nn.ReLU,
            weights_initializer=torch.nn.init.xavier_normal_,
            biases_initializer=None,
            output_nonlin=torch.nn.Tanh)

        self.expl_noise = 0.0
        super(Actor, self).__init__(pol_model, max_action)

    def forward(self, x, **kwargs):
        u = super(Actor, self).forward(x, **kwargs)
        if self.expl_noise > 0:
            if isinstance(x, np.ndarray):
                noise = self.expl_noise * np.random.randn(u.shape())
            else:
                noise = self.expl_noise * torch.randn_like(u)
            u = u + noise
        return u


class Critic(models.Regressor):
    def __init__(self, state_dim, action_dim, critic_hidden=[200] * 2):
        critic_model = models.mlp(
            state_dim + action_dim,
            1,
            critic_hidden,
            dropout_layers=[
                models.modules.CDropout(0.1)
                for i in range(len(critic_hidden))
            ],
            nonlin=torch.nn.ReLU,
            weights_initializer=torch.nn.init.xavier_normal_,
            biases_initializer=None)
        super(Critic, self).__init__(critic_model, None)


class DynModel(models.DynamicsModel):
    def __init__(self,
                 state_dim,
                 action_dim,
                 reward_func=None,
                 dyn_components=1,
                 dyn_hidden=[200] * 2):
        self.learn_reward = reward_func is None
        dynE = 2 * (state_dim + 1) if self.learn_reward else 2 * state_dim
        if dyn_components > 1:
            output_density = models.MixtureDensity(dynE / 2, dyn_components)
            dynE = (dynE + 1) * dyn_components
            self.log_likelihood_loss = losses.gaussian_mixture_log_likelihood
        else:
            output_density = models.DiagGaussianDensity(dynE / 2)
            self.log_likelihood_loss = losses.gaussian_log_likelihood

        dyn_model = models.mlp(
            state_dim + action_dim,
            dynE,
            dyn_hidden,
            dropout_layers=[
                models.modules.CDropout(0.5, 0.1)
                for i in range(len(dyn_hidden))
            ],
            nonlin=torch.nn.ReLU)
        super(DynModel, self).__init__(
            dyn_model, reward_func=reward_func, output_density=output_density)

    def fit(self,
            experience_dataset,
            batch_size=100,
            iterations=2000,
            optimizer=None):
        X, Y = experience_dataset.get_dynmodel_dataset(
            deltas=True, return_costs=self.learn_reward)
        self.set_dataset(
            torch.tensor(X).to(self.X.device, self.X.dtype),
            torch.tensor(Y).to(self.X.device, self.X.dtype))
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters())
        utils.train_regressor(
            self,
            iterations,
            batch_size,
            optimizer=optimizer,
            log_likelihood=self.log_likelihood_loss)


class MBDDPG(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, **kwargs):
        super(MBDDPG, self).__init__()
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        # initialize dynamics model
        self.dyn = DynModel(state_dim, action_dim, **kwargs)
        self.dyn_optimizer = torch.optim.Adam(self.dyn.parameters(), 1e-3)

    def forward(self, state, **kwargs):
        state = torch.tensor(state.reshape(1, -1)).float()
        return self.actor(state).cpu().data.numpy().flatten()

    def fit(self,
            experience_dataset,
            horizon,
            iterations,
            model_fit_iters=2000,
            batch_size=100,
            discount=0.99,
            tau=0.005):

        # train dynamics model with experience dataset
        self.dyn.fit(experience_dataset, batch_size)

        pbar = tqdm.tqdm(range(iterations))
        for it in pbar:
            # sample initial states for rollouts
            x0 = torch.tensor(
                experience_dataset.sample_states(batch_size, timestep=0)).to(
                    self.dyn.X.device).float()
            x0 = x0 + 1e-1 * x0.std(0) * torch.randn_like(x0)
            x0 = x0.detach()

            # Sample rollouts for ddpg updates
            self.actor.expl_noise = 1.0
            trajs = utils.rollout(x0, self.dyn, self.actor, horizon)
            self.actor.expl_noise = 0.0
            state, action, reward = (torch.stack(x).transpose(0, 1).detach()
                                     for x in zip(*trajs))
            state, next_state = (state[:-1].detach().reshape(
                -1, state.shape[-1]), state[1:].detach().reshape(
                    -1, state.shape[-1]))
            action = action[:-1].detach().reshape(-1, action.shape[-1])
            reward = reward[:-1].detach().reshape(-1, reward.shape[-1])

            # shuffle
            indices = list(range(state.shape[0]))
            random.shuffle(indices)
            N = state.shape[0]

            # update actor and critic with all rollout data
            self.actor.train()
            self.critic.train()
            self.actor_target.train()
            self.critic_target.train()
            for j in range(0, len(indices), batch_size):
                idx = indices[j:j + batch_size]
                state_ = state[idx]
                reward_ = reward[idx]
                next_state_ = next_state[idx]
                action_ = action[idx]
                # Compute the target Q value
                target_Q = self.critic_target(
                    torch.cat([next_state_,
                               self.actor_target(next_state_)], -1))

                target_Q = reward_ + discount * target_Q.detach()

                # Get current Q estimate
                current_Q = self.critic(torch.cat([state_, action_], -1))

                # Compute critic loss
                critic_loss = F.mse_loss(
                    current_Q,
                    target_Q) + self.critic.regularization_loss() / N

                # Optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Compute actor loss
                actor_loss = -self.critic(
                    torch.cat([state_, self.actor(state_)], -1)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                pbar.set_description("Actor loss: %f, Critic loss: %f" %
                                     (actor_loss, critic_loss))

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data +
                                        (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data +
                                        (1 - tau) * target_param.data)

            self.actor.eval()
            self.critic.eval()
            self.actor_target.eval()
            self.critic_target.eval()

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(),
                   '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(),
                   '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(
            torch.load('%s/%s_critic.pth' % (directory, filename)))
