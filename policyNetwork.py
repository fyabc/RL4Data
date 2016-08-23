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
                 input_size=ParamConfig['cnn_output_size'],
                 output_size=ParamConfig['train_batch_size']):
        self.input_size = input_size
        self.output_size = output_size

        self.W = theano.shared(name='W', value=init_norm(output_size, input_size))
        self.b = theano.shared(name='b', value=init_norm(output_size))
        self.parameters = [self.W, self.b]

        self.input = T.vector(name='softmax_probabilities')
        self.h = T.nnet.sigmoid(T.dot(self.W, self.input) + self.b)

        self.output_function = theano.function(
            inputs=[self.input],
            outputs=self.h,
        )

    def sample(self, output_prob):
        return np.random.random(self.output_size) < output_prob

    def update(self):
        pass


def test():
    pn = PolicyNetwork()

    input_data = np.ones(shape=(ParamConfig['cnn_output_size'],), dtype=fX)

    print(pn.output_function(input_data))


if __name__ == '__main__':
    test()
