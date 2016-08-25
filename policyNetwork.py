#! /usr/bin/python

from __future__ import print_function, unicode_literals

import numpy as np
import theano
import theano.tensor as T

from config import ParamConfig
from utils import fX, init_norm

__author__ = 'fyabc'


class PolicyNetwork(object):

    """
    The policy network.
    Input the softmax probabilities of output layer of NN, output the probabilities.
    Just use logistic regression.
    """

    def __init__(self,
                 input_size=ParamConfig['cnn_output_size']):
        self.input_size = input_size

        self.W = theano.shared(name='W', value=init_norm(input_size))
        self.b = theano.shared(name='b', value=init_norm())
        self.parameters = [self.W, self.b]

        # a minibatch of probabilities
        self.state = T.matrix(name='softmax_probabilities', dtype=fX)
        self.h = T.nnet.sigmoid(T.dot(self.state, self.W) + self.b)

        self.output_function = theano.function(
            inputs=[self.state],
            outputs=self.h,
        )

        self.reward = T.scalar('reward', dtype=fX)

    def take_action(self, state):
        return np.random.random(state.shape[0]) < self.output_function(state)

    def update(self):
        pass


def test():
    pn = PolicyNetwork()

    input_data = np.ones(shape=(4, ParamConfig['cnn_output_size']), dtype=fX)

    print(pn.take_action(input_data))


if __name__ == '__main__':
    test()
