# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import math
from collections import OrderedDict
import numpy as np

import theano.tensor as T
import theano
from lasagne.layers import LSTMLayer, InputLayer
from lasagne.layers import get_output, get_all_params

from utils import fX, floatX
from optimizers import adam

__author__ = 'fyabc'


class CriticNetwork(object):
    def __init__(self, feature_size, batch_size):
        self.feature_size = feature_size
        self.batch_size = batch_size

        self.action_ph = T.vector(dtype=fX, name='action')      # Shape = [batch_size]
        self.state_ph = T.matrix(dtype=fX, name='state')        # Shape = [batch_size, n_features]
        self.label = T.scalar(dtype=fX, name='label')

        self.inner_weights = theano.shared(
            np.random.normal(0.0, 1.0 / math.sqrt(feature_size * batch_size), (feature_size, batch_size)).astype(fX),
            name='inner_weights',
        )

        self.weights = theano.shared(
            np.random.normal(0.0, 1.0 / math.sqrt(batch_size), (batch_size,)).astype(fX),
            name='weights',
        )
        self.bias = theano.shared(floatX(0.0), name='bias')

        # Make output function
        self.output = T.nnet.relu(
            T.dot(self.weights, T.dot(T.dot(self.state_ph, self.inner_weights), self.action_ph)) + self.bias
        )

        self.parameters = [self.inner_weights, self.weights, self.bias]

        self.theta = OrderedDict()
        for parameter in self.parameters:
            self.theta[parameter.name] = parameter

        self.Q_function = theano.function([self.state_ph, self.action_ph], self.output)

        # Make update function
        loss = T.square(self.output - self.label).sum()
        grads = T.grad(loss, list(self.theta.values()))

        lr = T.scalar(dtype=fX)
        self.f_grad_shared, self.f_update = adam(
            lr, self.theta, grads, [self.state_ph, self.action_ph, self.label], loss)

    def update(self, state, action, label):
        loss = self.f_grad_shared(state, action, label)
        self.f_update(0.01)

        return loss


def test():
    B = 16
    F = 7

    theano.config.exception_verbosity = 'high'

    critic = CriticNetwork(F, B)

    state = np.zeros((B, F), dtype=fX)
    action = np.zeros((B,), dtype=fX)
    label = floatX(0.)

    print(critic.Q_function(state, action))
    print(critic.update(state, action, label))


if __name__ == '__main__':
    test()
