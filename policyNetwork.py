#! /usr/bin/python

from __future__ import print_function, unicode_literals

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from config import Config, ParamConfig
from utils import fX, floatX, init_norm, logging

__author__ = 'fyabc'


class PolicyNetwork(object):

    """
    The policy network.
    Input the softmax probabilities of output layer of NN, output the data selection result.
    Just use logistic regression.

    Parameters:
        input_size: the input size of the policy, should be the size of softmax probabilities of CNN.
    """

    @logging
    def __init__(self,
                 input_size=ParamConfig['cnn_output_size'],
                 optimizer=ParamConfig['policy_optimizer'],
                 learning_rate=ParamConfig['policy_learning_rate'],
                 gamma=ParamConfig['gamma'],
                 rb_update_rate=ParamConfig['reward_baseline_update_rate']
                 ):

        theano.config.exception_verbosity = 'high'

        self.input_size = input_size
        self.learning_rate = theano.shared(floatX(learning_rate), name='learning_rate')
        self.gamma = gamma
        self.rb_update_rate = rb_update_rate
        self.random_generator = RandomStreams(Config['seed'])

        # parameters to be learned
        self.W = theano.shared(name='W', value=init_norm(input_size))
        self.b = theano.shared(name='b', value=floatX(0.))
        self.parameters = [self.W, self.b]

        # a single case of input softmax probabilities
        self.input = T.vector(name='softmax_probabilities', dtype=fX)

        # build computation graph of output
        self.output = T.nnet.sigmoid(T.dot(self.input, self.W) + self.b)
        self.output_sample = self.random_generator.binomial(size=self.output.shape, p=self.output)

        self.output_sample_function = theano.function(
            inputs=[self.input],
            outputs=[self.output_sample, self.output_sample - self.output],
        )

        # replay buffers
        self.input_buffer = []
        self.action_buffer = []
        self.delta_buffer = []

        # the reward
        self.reward = T.scalar('reward', dtype=fX)
        # the baseline of the reward
        self.reward_baseline = 0.

        self.delta = T.scalar('delta', dtype=fX)

        grads = T.grad(
            T.log(self.output) * self.reward,
            self.parameters,
            known_grads={
                # TODO
                self.output: (self.output_sample - self.output)
            }
        )

        # TODO
        # optimizer
        updates = [(parameter, parameter + self.learning_rate * grad)
                   for parameter, grad in zip(self.parameters, grads)]

        self.update_function = theano.function(
            inputs=[self.input, self.reward],
            outputs=None,
            updates=updates,
            allow_input_downcast=True,
            on_unused_input='ignore',
        )

    def take_action(self, inputs):
        actions = np.zeros(shape=(inputs.shape[0],), dtype=bool)

        for i, input_ in enumerate(inputs):
            action, delta = self.output_sample_function(input_)
            action = bool(action)
            actions[i] = action

            self.input_buffer.append(input_)
            self.action_buffer.append(action)
            self.delta_buffer.append(delta)
        return actions

    @logging
    def update(self, reward):
        # TODO preprocess of reward

        # get discounted rewards
        discounted_rewards = np.zeros_like(self.action_buffer, dtype=fX)
        temp = reward - self.reward_baseline
        for i in reversed(xrange(discounted_rewards.size)):
            discounted_rewards[i] = temp
            temp *= self.gamma

        # TODO
        # update parameters for every time step
        for input_, r in zip(self.input_buffer, discounted_rewards):
            self.update_function(input_, r)

        # clear buffers
        self.input_buffer = []
        self.action_buffer = []
        self.delta_buffer = []

        # update reward baseline
        self.reward_baseline = (1 - self.rb_update_rate) * self.reward_baseline + self.rb_update_rate * reward

        print('New parameters:')
        print('$    w =', self.W.get_value())
        print('$    b =', self.b.get_value())
        print('New reward baseline:', self.reward_baseline)


def test():
    pn = PolicyNetwork()

    input_data = np.ones(shape=(4, ParamConfig['cnn_output_size']), dtype=fX)

    print(pn.take_action(input_data))


if __name__ == '__main__':
    test()
