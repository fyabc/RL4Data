# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

from collections import deque
import random
import numpy as np
import numpy.random as nr

from config import PolicyConfig

__author__ = 'fyabc'


class OUNoise(object):
    """Ornstein-Uhlenbeck Noise"""
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

    @staticmethod
    def test():
        ou = OUNoise(3)
        states = []
        for i in range(1000):
            states.append(ou.noise())
        import matplotlib.pyplot as plt

        plt.plot(states)
        plt.show()


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        return random.sample(self.buffer, batch_size)

    def maxlen(self):
        return self.buffer.maxlen

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        self.buffer.append(experience)

    def clear(self):
        self.buffer.clear()


class ActorNetwork(object):
    pass


class CriticNetwork(object):
    pass


class DDPG(object):
    def __init__(self, environment):
        self.environment = environment

        self.state_dim = None
        self.action_dim = 2

        # Randomly initialize actor network and critic network
        # with both their target networks
        self.actor_network = None
        self.critic_network = None

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(PolicyConfig['replay_buffer_size'])

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)

    def train(self):
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(PolicyConfig['DDPG_batch_size'])

        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

    def noise_action(self, state):
        pass

    def action(self, state):
        pass

    def perceive(self, state, action, reward, next_state, done):
        pass


def test():
    OUNoise.test()


if __name__ == '__main__':
    test()
