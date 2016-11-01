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
from lasagne.nonlinearities import softmax, rectify, tanh
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

    def get_test_acc(self, x_test, y_test):
        test_loss, test_acc, test_batches = self.validate_or_test(x_test, y_test)
        return test_acc / test_batches

    def reset_learning_rate(self):
        pass

    @staticmethod
    def get_policy_input_size():
        input_size = MNISTModel.output_size
        if PolicyConfig['add_label_input']:
            input_size += 1
        if PolicyConfig['add_label']:
            # input_size += 1
            input_size += MNISTModel.output_size
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
        batch_size = targets.shape[0]

        probability = self.f_probs(inputs)

        if PolicyConfig['add_label_input']:
            label_inputs = np.zeros(shape=(batch_size, 1), dtype=fX)
            for i in range(batch_size):
                # assert probability[i, targets[i]] > 0, 'Probability <= 0!!!'
                label_inputs[i, 0] = np.log(max(probability[i, targets[i]], 1e-9))
            probability = np.hstack([probability, label_inputs])

        if PolicyConfig['add_label']:
            # labels = floatX(targets) * (1.0 / ParamConfig['cnn_output_size'])
            # probability = np.hstack([probability, labels[:, None]])

            labels = np.zeros(shape=(batch_size, self.output_size), dtype=fX)
            for i, target in enumerate(targets):
                labels[i, target] = 1.

            probability = np.hstack([probability, labels])

        if PolicyConfig['use_first_layer_output']:
            first_layer_output = self.f_first_layer_output(inputs)
            shape_first = np.product(first_layer_output.shape[1:])
            first_layer_output = first_layer_output.reshape((batch_size, shape_first))
            probability = np.hstack([probability, first_layer_output])

        if PolicyConfig['add_epoch_number']:
            epoch_number_inputs = np.full((batch_size, 1), floatX(epoch) / ParamConfig['epoch_per_episode'], dtype=fX)
            probability = np.hstack([probability, epoch_number_inputs])

        if PolicyConfig['add_learning_rate']:
            learning_rate_inputs = np.full((batch_size, 1), self.learning_rate.get_value(), dtype=fX)
            probability = np.hstack([probability, learning_rate_inputs])

        if PolicyConfig['add_margin']:
            margin_inputs = np.zeros(shape=(batch_size, 1), dtype=fX)
            for i in range(batch_size):
                prob_i = probability[i].copy()
                margin_inputs[i, 0] = prob_i[targets[i]]
                prob_i[targets[i]] = -np.inf

                margin_inputs[i, 0] -= np.max(prob_i)
            probability = np.hstack([probability, margin_inputs])

        if PolicyConfig['add_average_accuracy']:
            avg_acc_inputs = np.full((batch_size, 1), average(history_accuracy), dtype=fX)
            probability = np.hstack([probability, avg_acc_inputs])

        return probability


class MNISTModel(MNISTModelBase):
    def __init__(self,
                 hidden_size=None,
                 train_batch_size=None,
                 validate_batch_size=None):
        super(MNISTModel, self).__init__(train_batch_size, validate_batch_size)
        self.hidden_size = hidden_size or ParamConfig['hidden_size']

        self.learning_rate = theano.shared(floatX(ParamConfig['learning_rate']))

        # Prepare Theano variables for inputs and targets
        self.input_var = T.matrix('inputs')
        self.target_var = T.vector('targets', dtype='int64')

        self.network = self.build_cnn()
        print("number of parameters in model: %d" % lasagne.layers.count_params(self.network, trainable=True))

        self.saved_init_parameters_values = get_all_param_values(self.network, trainable=True)

        self.build_train_function()
        self.build_validate_function()

    def build_cnn(self):
        # Building the network
        layer_in = InputLayer(shape=(None, 784), input_var=self.input_var)

        # Hidden layer
        layer = DenseLayer(
            layer_in,
            num_units=self.hidden_size,
            W=lasagne.init.Uniform(
                range=(-np.sqrt(6. / (784 + self.hidden_size)),
                       np.sqrt(6. / (784 + self.hidden_size)))),
            nonlinearity=tanh,
        )

        # LR layer
        layer = DenseLayer(
            layer,
            num_units=self.output_size,
            W=lasagne.init.Constant(0.),
            nonlinearity=softmax,
        )

        return layer

    def build_train_function(self):
        probs = lasagne.layers.get_output(self.network)
        self.f_probs = theano.function(
            inputs=[self.input_var],
            outputs=probs
        )

        loss = lasagne.objectives.categorical_crossentropy(probs, self.target_var)

        self.f_cost_list_without_decay = theano.function([self.input_var, self.target_var], loss)

        loss = loss.mean()

        self.f_cost_without_decay = theano.function([self.input_var, self.target_var], loss)

        # add weight decay
        all_layers = lasagne.layers.get_all_layers(self.network)
        l1_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l1) *\
            ParamConfig['l1_penalty_factor']
        loss += l1_penalty
        l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) *\
            ParamConfig['l2_penalty_factor']
        loss += l2_penalty

        self.f_cost = theano.function([self.input_var, self.target_var], loss)

        params = lasagne.layers.get_all_params(self.network, trainable=True)

        # SGD update.
        updates_sgd = lasagne.updates.sgd(loss, params, self.learning_rate)

        f_train_sgd = theano.function([self.input_var, self.target_var], loss, updates=updates_sgd)
        self.f_train = f_train_sgd

    def build_validate_function(self):
        test_preds = lasagne.layers.get_output(self.network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_preds, self.target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_preds, axis=1), self.target_var), dtype=fX)

        self.f_validate = theano.function([self.input_var, self.target_var], [test_loss, test_acc])


if __name__ == '__main__':
    pass
