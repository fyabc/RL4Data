#! /usr/bin/python

from __future__ import print_function, unicode_literals

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from config import Config, ParamConfig
from utils import fX, init_norm

__author__ = 'fyabc'


class PolicyNetwork(object):

    """
    The policy network.
    Input the softmax probabilities of output layer of NN, output the data selection result.
    Just use logistic regression.

    Parameters:
        input_size: the input size of the policy, should be the size of softmax probabilities of CNN.
    """

    def __init__(self,
                 input_size=ParamConfig['cnn_output_size'],
                 optimizer=ParamConfig['policy_optimizer'],
                 learning_rate=ParamConfig['policy_learning_rate']):

        self.input_size = input_size
        self.learning_rate = learning_rate
        self.random_generator = RandomStreams(Config['seed'])

        # parameters to be learned
        self.W = theano.shared(name='W', value=init_norm(input_size))
        self.b = theano.shared(name='b', value=init_norm())
        self.parameters = {
            'W': self.W,
            'b': self.b,
        }

        # a minibatch of input softmax probabilities
        self.input = T.matrix(name='softmax_probabilities', dtype=fX)

        # build computation graph of output
        self.output = T.nnet.sigmoid(T.dot(self.input, self.W) + self.b)
        self.output_sample = self.random_generator.binomial(size=self.output.shape, p=self.output)

        self.output_function = theano.function(
            inputs=[self.input],
            outputs=self.output,
        )

        self.output_sample_function = theano.function(
            inputs=[self.input],
            outputs=self.output_sample,
        )

        # replay buffers
        self.input_buffer = []
        self.action_buffer = []

        self.reward = T.scalar('reward', dtype=fX)

        # TODO
        # cost
        self.cost = None

        grads = T.grad(
            self.cost,
            self.parameters.items(),
            known_grads={
                # TODO
                self.output: None
            }
        )

        # TODO
        # optimizer

    def take_action(self, input_):
        actions = np.asarray(self.output_sample_function(input_), dtype='bool')
        self.input_buffer.append(input_)
        self.action_buffer.append(actions)
        return actions

    def update(self, reward):
        # TODO

        # clear buffers
        self.input_buffer = []
        self.action_buffer = []


def test():
    pn = PolicyNetwork()

    input_data = np.ones(shape=(4, ParamConfig['cnn_output_size']), dtype=fX)

    print(pn.take_action(input_data))


if __name__ == '__main__':
    test()
