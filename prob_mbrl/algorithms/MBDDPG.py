"""
    Model based DDPG based on the code available here: https://github.com/sfujim/TD3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from prob_mbrl import models, losses, utils

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class MBDDPG(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 reward_func=None,
                 dyn_components=1,
                 dyn_hidden=[200] * 2):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        # initialize dynamics model
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
        self.dyn = models.DynamicsModel(
            dyn_model, reward_func=reward_func,
            output_density=output_density).float()
        self.dyn_optimizer = torch.optim.Adam(self.dyn.parameters())

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self,
              experience_dataset,
              horizon,
              iterations,
              batch_size=100,
              discount=0.99,
              tau=0.005):

        # train dynamics model with experience dataset
        X, Y = experience_dataset.get_dynmodel_dataset(
            deltas=True, return_costs=self.learn_reward)
        self.dyn.set_dataset(
            torch.tensor(X).to(self.dyn.X.device).float(),
            torch.tensor(Y).to(self.dyn.X.device).float())
        utils.train_regressor(
            self.dyn,
            2000,
            batch_size,
            True,
            self.dyn_optimizer,
            log_likelihood=self.log_likelihood_loss)

        for it in range(iterations):
            # sample initial states for rollouts
            x0 = torch.tensor(exp.sample_states(N_particles, timestep=0)).to(
                dyn.X.device).float()
            x0 = x0 + 1e-1 * x0.std(0) * torch.randn_like(x0)
            x0 = x0.detach()
            # Sample rollouts for critic
            trajs = utils.core.rollout(
                x0,
                self.dyn,
                self.actor,
                iterations,
                resample_model=False,
                resample_policy=False,
                resample_particles=False)
            state, action, reward = (torch.stack(x).transpose(0, 1).detach()
                                     for x in zip(*trajs))
            state, next_state = state[:-1], state[1:]
            action = action[:-1]
            reward = reward[:-1]

            # Compute the target Q value
            target_Q = self.critic_target(next_state,
                                          self.actor_target(next_state))
            target_Q = reward + (discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data +
                                        (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data +
                                        (1 - tau) * target_param.data)

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
