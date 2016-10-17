# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import numpy as np

import theano.tensor as T
import theano
from lasagne.layers import LSTMLayer, InputLayer, DenseLayer
from lasagne.layers import get_output, get_all_params

from utils import fX

__author__ = 'fyabc'


class ActorNetwork(object):
    def __init__(self, hidden_size, layer_num, unroll, n_dimension, batch_size=1, update_batch_size=10):
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.unroll = unroll

        self._lr = T.scalar(dtype=fX)
        self._input = T.matrix(dtype=fX)

        self._state = theano.shared(value=np.zeros((batch_size, 4 * self.hidden_size), dtype=fX), name='state')
        _state = self._state

        layer = InputLayer(shape=(), input_var=self._input)

        for _ in range(self.layer_num):
            layer = LSTMLayer(layer, num_units=self.hidden_size)

        self.lstm_output = get_output(layer)

        self.softmax_W = theano.shared(np.zeros((hidden_size, 1), dtype=fX), name='softmax_W')
        self.softmax_b = theano.shared(np.zeros((1,), dtype=fX), name='softmax_b')

        for _ in range(self.unroll):
            # Unroll
            pass

        self.lstm_variables = get_all_params(layer)

    def take_action(self, loss):
        pass


class CriticNetwork(object):
    pass
