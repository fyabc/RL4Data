#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import time
import numpy as np
import theano
import theano.tensor as T

import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm
from lasagne.layers.helper import get_all_param_values, set_all_param_values

# For vanilia model
from lasagne.layers import LocalResponseNormalization2DLayer, MaxPool2DLayer

from config import Config, MNISTConfig as ParamConfig, PolicyConfig
from utils import logging, iterate_minibatches, fX, floatX, shuffle_data, average, message


class MNISTModelBase(object):

    output_size = 10

    def __init__(self,
                 train_batch_size=None,
                 validate_batch_size=None):
        # Functions and parameters that must be provided.
        self.train_batch_size = train_batch_size or ParamConfig['train_batch_size']
        self.validate_batch_size = validate_batch_size or ParamConfig['validate_batch_size']

        self.network = None
        self.saved_init_parameters_values = None

        self.learning_rate = None

        self.f_first_layer_output = None
        self.f_probs = None
        self.f_cost_list_without_decay = None
        self.f_train = None
        self.f_alpha_train = None
        self.f_cost_without_decay = None
        self.f_cost = None

        self.f_train = None
        self.f_validate = None

    def build_train_function(self):
        pass

    def build_validate_function(self):
        pass

    def reset_parameters(self):
        set_all_param_values(self.network, self.saved_init_parameters_values, trainable=True)

    @logging
    def save_model(self, filename=None):
        filename = filename or Config['model_file']
        np.savez(filename, *get_all_param_values(self.network))

    @logging
    def load_model(self, filename=None):
        filename = filename or Config['model_file']
        with np.load(filename) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.network, param_values)

    def get_training_loss(self, x_train, y_train):
        sum_loss = 0.0
        training_batches = 0
        for batch in iterate_minibatches(x_train, y_train, self.train_batch_size, shuffle=False, augment=False):
            training_batches += 1
            inputs, targets = batch
            sum_loss += self.f_cost(inputs, targets)
        return sum_loss / training_batches

    @logging
    def test(self, x_test, y_test):
        test_err, test_acc, test_batches = self.validate_or_test(x_test, y_test)
        message("$Final results:")
        message("$  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        message("$  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))

    def validate_or_test(self, x_test, y_test):
        # Calculate validation error of model:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(x_test, y_test, self.validate_batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = self.f_validate(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        return test_err, test_acc, test_batches

    def reset_learning_rate(self):
        pass

    @staticmethod
    def get_policy_input_size():
        input_size = ParamConfig['cnn_output_size']
        if PolicyConfig['add_label_input']:
            input_size += 1
        if PolicyConfig['add_label']:
            # input_size += 1
            input_size += ParamConfig['cnn_output_size']
        if PolicyConfig['use_first_layer_output']:
            input_size += 16 * 32 * 32
        if PolicyConfig['add_epoch_number']:
            input_size += 1
        if PolicyConfig['add_learning_rate']:
            input_size += 1
        if PolicyConfig['add_margin']:
            input_size += 1
        if PolicyConfig['add_average_accuracy']:
            input_size += 1
        return input_size

    def get_policy_input(self, inputs, targets, epoch, history_accuracy=None):


class MNISTModel(MNISTModelBase):
    def __init__(self,
                 train_batch_size=None,
                 validate_batch_size=None):
        super(MNISTModel, self).__init__(train_batch_size, validate_batch_size)



if __name__ == '__main__':
    pass
