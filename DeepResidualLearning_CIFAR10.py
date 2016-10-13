#! /usr/bin/python

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

from config import Config, CifarConfig, PolicyConfig
from utils import logging, iterate_minibatches, fX, floatX, shuffle_data, average


class CNN(object):
    """
    The neural network model.
    """

    output_size = CifarConfig['cnn_output_size']

    def __init__(self, n=None):
        n = n or CifarConfig['n']

        self.train_batch_size = CifarConfig['train_batch_size']
        self.validate_batch_size = CifarConfig['validate_batch_size']

        self.learning_rate = theano.shared(lasagne.utils.floatX(CifarConfig['init_learning_rate']))

        # Prepare Theano variables for inputs and targets
        self.input_var = T.tensor4('inputs')
        self.target_var = T.ivector('targets')

        self.network = self.build_cnn(self.input_var, n)
        print("number of parameters in model: %d" % lasagne.layers.count_params(self.network, trainable=True))

        self.saved_init_parameters_values = get_all_param_values(self.network, trainable=True)

        self.probs_function = None
        self.train_function = None
        self.validate_function = None

        self.alpha_train_function = None

        self.build_train_function()
        self.build_validate_function()

    @logging
    def build_cnn(self, input_var=None, n=CifarConfig['n']):
        # create a residual learning building block with two stacked 3x3 conv-layers as in paper
        def residual_block(layer_, increase_dim=False, projection=False):
            input_num_filters = layer_.output_shape[1]
            if increase_dim:
                first_stride = (2, 2)
                out_num_filters = input_num_filters * 2
            else:
                first_stride = (1, 1)
                out_num_filters = input_num_filters

            stack_1 = batch_norm(
                    ConvLayer(layer_, num_filters=out_num_filters, filter_size=(3, 3), stride=first_stride,
                              nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'),
                              flip_filters=False))
            stack_2 = batch_norm(
                    ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3, 3), stride=(1, 1),
                              nonlinearity=None,
                              pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

            # add shortcut connections
            if increase_dim:
                if projection:
                    # projection shortcut, as option B in paper
                    projection = batch_norm(
                            ConvLayer(layer_, num_filters=out_num_filters, filter_size=(1, 1), stride=(2, 2),
                                      nonlinearity=None, pad='same', b=None, flip_filters=False))
                    block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]), nonlinearity=rectify)
                else:
                    # identity shortcut, as option A in paper
                    identity = ExpressionLayer(layer_, lambda X: X[:, :, ::2, ::2],
                                               lambda s: (s[0], s[1], s[2] // 2, s[3] // 2))
                    padding = PadLayer(identity, [out_num_filters // 4, 0, 0], batch_ndim=1)
                    block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]), nonlinearity=rectify)
            else:
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, layer_]), nonlinearity=rectify)

            return block

        # Building the network
        layer_in = InputLayer(shape=(None, 3, 32, 32), input_var=input_var)

        # first layer, output is 16 x 32 x 32
        layer = batch_norm(ConvLayer(layer_in, num_filters=16, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify,
                                     pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        self.first_layer_output = lasagne.layers.get_output(layer, inputs=input_var)
        self.first_layer_output_function = theano.function(
            inputs=[input_var],
            outputs=self.first_layer_output
        )

        # first stack of residual blocks, output is 16 x 32 x 32
        for _ in range(n):
            layer = residual_block(layer)

        # second stack of residual blocks, output is 32 x 16 x 16
        layer = residual_block(layer, increase_dim=True)
        for _ in range(1, n):
            layer = residual_block(layer)

        # third stack of residual blocks, output is 64 x 8 x 8
        layer = residual_block(layer, increase_dim=True)
        for _ in range(1, n):
            layer = residual_block(layer)

        # average pooling
        layer = GlobalPoolLayer(layer)

        # fully connected layer
        return DenseLayer(
                layer, num_units=10,
                W=lasagne.init.HeNormal(),
                nonlinearity=softmax)

    @logging
    def build_train_function(self):
        """build train functions"""

        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        probs = lasagne.layers.get_output(self.network)
        self.probs_function = theano.function(
            inputs=[self.input_var],
            outputs=probs
        )

        loss = lasagne.objectives.categorical_crossentropy(probs, self.target_var)
        loss = loss.mean()

        # add weight decay
        all_layers = lasagne.layers.get_all_layers(self.network)
        l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * \
                     CifarConfig['l2_penalty_factor']
        loss += l2_penalty

        # Create update expressions for training
        # Stochastic Gradient Descent (SGD) with momentum
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.momentum(
                loss, params, learning_rate=self.learning_rate, momentum=CifarConfig['momentum'])

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        self.train_function = theano.function([self.input_var, self.target_var], loss, updates=updates)

        # ##########################################################

        # build train functions, which loss is `\alpha \dot orig_loss`.
        # \alpha is the weights of each input of one minibatch. \sigma_{minibatch}(alpha) = 1.
        # \alpha is given by the policy network.
        alpha = T.vector('alpha', dtype=fX)

        alpha_loss = lasagne.objectives.categorical_crossentropy(probs, self.target_var)
        alpha_loss = T.dot(alpha_loss, alpha)

        alpha_loss += l2_penalty
        updates = lasagne.updates.momentum(
                alpha_loss, params, learning_rate=self.learning_rate, momentum=CifarConfig['momentum'])

        self.alpha_train_function = theano.function(
                [self.input_var, self.target_var, alpha], alpha_loss, updates=updates)

    @logging
    def build_validate_function(self):
        """build validate functions"""

        # Create a loss expression for validation/testing
        test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                self.target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), self.target_var),
                          dtype=theano.config.floatX)

        # Compile a second function computing the validation loss and accuracy:
        self.validate_function = theano.function([self.input_var, self.target_var], [test_loss, test_acc])

    @logging
    def train(self, x_train, y_train, x_test, y_test, num_epochs):
        if self.train_function is None:
            self.build_train_function()

        # launch the training loop
        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(num_epochs):
            print('Epoch {} of {}:'.format(epoch, num_epochs))

            # shuffle training data
            x_train, y_train = shuffle_data(x_train, y_train)

            start_time = time.time()

            # Train one epoch:
            train_err, train_batches, softmax_probabilities = self.train_one_epoch(x_train, y_train)

            # And a full pass over the validation data:
            validate_err, validate_acc, validate_batches = self.validate_or_test(x_test, y_test)

            # Then we print the results for this epoch:
            print("This epoch took {:.3f}s".format(time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(validate_err / validate_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                validate_acc / validate_batches * 100))

            # Adjust learning rate as in paper
            # 32k and 48k iterations should be roughly equivalent to 41 and 61 epochs
            if (epoch + 1) == 41 or (epoch + 1) == 61:
                new_lr = self.learning_rate.get_value() * CifarConfig['learning_rate_discount']
                print("New LR:" + str(new_lr))
                self.learning_rate.set_value(lasagne.utils.floatX(new_lr))

        # dump the network weights to a file:
        np.savez(str('cifar10_deep_residual_model.npz'), *lasagne.layers.get_all_param_values(self.network))

    @logging
    def train_one_epoch(self, x_train, y_train):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        softmax_probabilities = []

        for batch in iterate_minibatches(x_train, y_train, self.train_batch_size, shuffle=True, augment=True):
            inputs, targets = batch

            train_err += self.train_function(inputs, targets)
            train_batches += 1

            current_prediction = self.probs_function(inputs)
            softmax_probabilities.append(current_prediction)

        return train_err, train_batches, np.vstack(softmax_probabilities)

    def train_one_minibatch(self, inputs, targets, mask=None):
        if mask is not None:
            inputs = inputs[mask]
            targets = targets[mask]

        return self.train_function(inputs, targets), self.probs_function(inputs)

    def reset_all_parameters(self):
        set_all_param_values(self.network, self.saved_init_parameters_values, trainable=True)

    @logging
    def update_learning_rate(self):
        self.learning_rate.set_value(floatX(self.learning_rate.get_value() * 0.1))

    @logging
    def reset_learning_rate(self):
        self.learning_rate.set_value(lasagne.utils.floatX(CifarConfig['init_learning_rate']))

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

    @logging
    def test(self, x_test, y_test):
        test_err, test_acc, test_batches = self.validate_or_test(x_test, y_test)
        print("$Final results:")
        print("$  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("$  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))

    # @logging
    def validate_or_test(self, x_test, y_test):
        # Calculate validation error of model:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(x_test, y_test, self.validate_batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = self.validate_function(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        return test_err, test_acc, test_batches

    @staticmethod
    def get_policy_input_size():
        input_size = CifarConfig['cnn_output_size']
        if PolicyConfig['add_label_input']:
            input_size += 1
        if PolicyConfig['add_label']:
            # input_size += 1
            input_size += CifarConfig['cnn_output_size']
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

        probability = self.probs_function(inputs)

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
            first_layer_output = self.first_layer_output_function(inputs)
            shape_first = np.product(first_layer_output.shape[1:])
            first_layer_output = first_layer_output.reshape((batch_size, shape_first))
            probability = np.hstack([probability, first_layer_output])

        if PolicyConfig['add_epoch_number']:
            epoch_number_inputs = np.full((batch_size, 1), floatX(epoch) / CifarConfig['epoch_per_episode'], dtype=fX)
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


def test():
    pass


if __name__ == '__main__':
    test()
