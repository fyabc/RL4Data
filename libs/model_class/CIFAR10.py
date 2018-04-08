#! /usr/bin/python

from __future__ import print_function

import time

import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import InputLayer
from lasagne.layers import LocalResponseNormalization2DLayer, MaxPool2DLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import PadLayer
from lasagne.layers import batch_norm
from lasagne.layers.helper import get_all_param_values, set_all_param_values
from lasagne.nonlinearities import softmax, rectify

from ..utility.CIFAR10 import iterate_minibatches
from ..utility.config import CifarConfig as ParamConfig, PolicyConfig
from ..utility.my_logging import message, logging
from ..utility.name_register import NameRegister
from ..utility.utils import fX, floatX, shuffle_data, average, get_rank


class CIFARModelBase(NameRegister):
    """
    The base class of CIFAR-10 network model.
    """

    NameTable = {}

    output_size = ParamConfig['cnn_output_size']

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
        filename = filename or ParamConfig['model_file']
        np.savez(filename, *get_all_param_values(self.network))

    @logging
    def load_model(self, filename=None):
        filename = filename or ParamConfig['model_file']
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
        input_size = 0
        if PolicyConfig['add_output']:
            input_size += CIFARModelBase.output_size
        if PolicyConfig['add_label_input']:
            input_size += 1
        if PolicyConfig['add_label']:
            input_size += CIFARModelBase.output_size
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
        if PolicyConfig['add_loss_rank']:
            input_size += 1
        if PolicyConfig['add_accepted_data_number']:
            input_size += 1
        return input_size

    def get_policy_input(self, inputs, targets, updater, history_accuracy=None):
        batch_size = targets.shape[0]

        to_be_stacked = []

        probability = self.f_probs(inputs)

        if PolicyConfig['add_output']:
            to_be_stacked.append(probability)

        if PolicyConfig['add_label_input']:
            label_inputs = np.zeros(shape=(batch_size, 1), dtype=fX)
            for i in range(batch_size):
                # assert probability[i, targets[i]] > 0, 'Probability <= 0!!!'
                label_inputs[i, 0] = np.log(max(probability[i, targets[i]], 1e-9))
            to_be_stacked.append(label_inputs)

        if PolicyConfig['add_label']:
            labels = np.zeros(shape=(batch_size, self.output_size), dtype=fX)
            for i, target in enumerate(targets):
                labels[i, target] = 1.

            to_be_stacked.append(labels)

        if PolicyConfig['use_first_layer_output']:
            first_layer_output = self.f_first_layer_output(inputs)
            shape_first = np.product(first_layer_output.shape[1:])
            first_layer_output = first_layer_output.reshape((batch_size, shape_first))
            to_be_stacked.append(first_layer_output)

        if PolicyConfig['add_epoch_number']:
            epoch_number_inputs = np.full((batch_size, 1),
                                          floatX(updater.epoch) / ParamConfig['epoch_per_episode'], dtype=fX)
            to_be_stacked.append(epoch_number_inputs)

        if PolicyConfig['add_learning_rate']:
            learning_rate_inputs = np.full((batch_size, 1), self.learning_rate.get_value(), dtype=fX)
            to_be_stacked.append(learning_rate_inputs)

        if PolicyConfig['add_margin']:
            margin_inputs = np.zeros(shape=(batch_size, 1), dtype=fX)
            for i in range(batch_size):
                prob_i = probability[i].copy()
                margin_inputs[i, 0] = prob_i[targets[i]]
                prob_i[targets[i]] = -np.inf

                margin_inputs[i, 0] -= np.max(prob_i)
            to_be_stacked.append(margin_inputs)

        if PolicyConfig['add_average_accuracy']:
            avg_acc_inputs = np.full((batch_size, 1), average(history_accuracy), dtype=fX)
            to_be_stacked.append(avg_acc_inputs)

        if PolicyConfig['add_loss_rank']:
            cost_list_without_decay = self.f_cost_list_without_decay(inputs, targets)
            rank = get_rank(cost_list_without_decay).astype(fX) / batch_size

            to_be_stacked.append(rank.reshape((batch_size, 1)))

        if PolicyConfig['add_accepted_data_number']:
            accepted_data_number_inputs = np.full(
                (batch_size, 1),
                updater.total_accepted_cases / (updater.data_size * ParamConfig['epoch_per_episode']),
                dtype=fX)

            to_be_stacked.append(accepted_data_number_inputs)

        return np.hstack(to_be_stacked)


class CIFARModel(CIFARModelBase):
    """
    The CIFAR-10 neural network model (ResNet).
    """

    def __init__(self,
                 n=None,
                 train_batch_size=None,
                 validate_batch_size=None):
        super(CIFARModel, self).__init__(train_batch_size, validate_batch_size)

        n = n or ParamConfig['n']

        self.learning_rate = theano.shared(lasagne.utils.floatX(ParamConfig['init_learning_rate']))

        # Prepare Theano variables for inputs and targets
        self.input_var = T.tensor4('inputs')
        self.target_var = T.ivector('targets')

        self.network = self.build_cnn(self.input_var, n)
        message("number of parameters in model: %d" % lasagne.layers.count_params(self.network, trainable=True))

        self.saved_init_parameters_values = get_all_param_values(self.network, trainable=True)

        self.build_train_function()
        self.build_validate_function()

    @logging
    def build_cnn(self, input_var=None, n=ParamConfig['n']):
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
        first_layer_output = lasagne.layers.get_output(layer, inputs=input_var)
        self.f_first_layer_output = theano.function(
            inputs=[input_var],
            outputs=first_layer_output
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
        l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * \
            ParamConfig['l2_penalty_factor']
        loss += l2_penalty

        self.f_cost = theano.function([self.input_var, self.target_var], loss)

        # Create update expressions for training
        # Stochastic Gradient Descent (SGD) with momentum
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.momentum(
            loss, params, learning_rate=self.learning_rate, momentum=ParamConfig['momentum'])

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        self.f_train = theano.function([self.input_var, self.target_var], loss, updates=updates)

        # ##########################################################

        # build train functions, which loss is `\alpha \dot orig_loss`.
        # \alpha is the weights of each input of one minibatch. \sigma_{minibatch}(alpha) = 1.
        # \alpha is given by the policy network.
        alpha = T.vector('alpha', dtype=fX)

        alpha_loss = lasagne.objectives.categorical_crossentropy(probs, self.target_var)
        alpha_loss = T.dot(alpha_loss, alpha)

        alpha_loss += l2_penalty
        updates = lasagne.updates.momentum(
            alpha_loss, params, learning_rate=self.learning_rate, momentum=ParamConfig['momentum'])

        self.f_alpha_train = theano.function(
            [self.input_var, self.target_var, alpha], alpha_loss, updates=updates)

    @logging
    def build_validate_function(self):
        """build validate functions"""

        # Create a loss expression for validation/testing
        test_preds = lasagne.layers.get_output(self.network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_preds,
                                                                self.target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_preds, axis=1), self.target_var),
                          dtype=theano.config.floatX)

        # Compile a second function computing the validation loss and accuracy:
        self.f_validate = theano.function([self.input_var, self.target_var], [test_loss, test_acc])

    @logging
    def train(self, x_train, y_train, x_test, y_test, num_epochs):
        """Stub function for old code. Will be removed in future."""

        if self.f_train is None:
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
                new_lr = self.learning_rate.get_value() * ParamConfig['learning_rate_discount']
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

            train_err += self.f_train(inputs, targets)
            train_batches += 1

            current_prediction = self.f_probs(inputs)
            softmax_probabilities.append(current_prediction)

        return train_err, train_batches, np.vstack(softmax_probabilities)

    def train_one_minibatch(self, inputs, targets, mask=None):
        if mask is not None:
            inputs = inputs[mask]
            targets = targets[mask]

        return self.f_train(inputs, targets), self.f_probs(inputs)

    @logging
    def update_learning_rate(self):
        self.learning_rate.set_value(floatX(self.learning_rate.get_value() * 0.1))

    @logging
    def reset_learning_rate(self):
        self.learning_rate.set_value(lasagne.utils.floatX(ParamConfig['init_learning_rate']))


CIFARModel.register_class(['resnet'])


class VanillaCNNModel(CIFARModelBase):
    """
    The CIFAR-10 neural network model (Vanilla CNN).
    """

    def __init__(self,
                 train_batch_size=None,
                 valid_batch_size=None
                 ):
        super(VanillaCNNModel, self).__init__(train_batch_size, valid_batch_size)

        # Prepare Theano variables for inputs and targets
        self.input_var = T.tensor4('inputs')
        self.target_var = T.ivector('targets')

        self.learning_rate = theano.shared(lasagne.utils.floatX(ParamConfig['init_learning_rate']))

        self.network = self.build_cnn(self.input_var)
        message("number of parameters in model: %d" % lasagne.layers.count_params(self.network, trainable=True))

        self.saved_init_parameters_values = get_all_param_values(self.network, trainable=True)

        self.build_train_function()
        self.build_validate_function()

    def build_cnn(self, input_var=None):
        # Building the network
        layer_in = InputLayer(shape=(None, 3, 32, 32), input_var=input_var)

        # Conv1
        # [NOTE]: normal vs. truncated normal?
        # [NOTE]: conv in lasagne is not same as it in TensorFlow.
        layer = ConvLayer(layer_in, num_filters=64, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify,
                          pad='same', W=lasagne.init.HeNormal(), flip_filters=False)
        # Pool1
        layer = MaxPool2DLayer(layer, pool_size=(3, 3), stride=(2, 2))
        # Norm1
        layer = LocalResponseNormalization2DLayer(layer, alpha=0.001 / 9.0, k=1.0, beta=0.75)

        # Conv2
        layer = ConvLayer(layer, num_filters=64, filter_size=(5, 5), stride=(1, 1), nonlinearity=rectify,
                          pad='same', W=lasagne.init.HeNormal(), flip_filters=False)
        # Norm2
        # [NOTE]: n must be odd, but n in Chang's code is 4?
        layer = LocalResponseNormalization2DLayer(layer, alpha=0.001 / 9.0, k=1.0, beta=0.75)
        # Pool2
        layer = MaxPool2DLayer(layer, pool_size=(3, 3), stride=(2, 2))

        # Reshape
        layer = lasagne.layers.ReshapeLayer(layer, shape=([0], -1))

        # Dense3
        layer = DenseLayer(layer, num_units=384, W=lasagne.init.HeNormal(), b=lasagne.init.Constant(0.1))

        # Dense4
        layer = DenseLayer(layer, num_units=192, W=lasagne.init.Normal(std=0.04), b=lasagne.init.Constant(0.1))

        # Softmax
        layer = DenseLayer(layer, num_units=self.output_size,
                           W=lasagne.init.Normal(std=1. / 192.0), nonlinearity=softmax)

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
        l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.004
        loss += l2_penalty

        self.f_cost = theano.function([self.input_var, self.target_var], loss)

        params = lasagne.layers.get_all_params(self.network, trainable=True)

        # Different updates.
        # updates_sgd = lasagne.updates.sgd(loss, params, self.learning_rate)
        # updates_momentum = lasagne.updates.momentum(
        #     loss, params, learning_rate=self.learning_rate, momentum=ParamConfig['momentum'])

        # [NOTE]: Some default values of lasagne and TensorFlow are same.
        updates_adam = lasagne.updates.adam(loss, params, learning_rate=0.001)

        # updates_adagrad = lasagne.updates.adagrad(loss, params, learning_rate=self.learning_rate)
        # updates_adadelta = lasagne.updates.adadelta(loss, params, learning_rate=0.001, epsilon=1e-8)
        # updates_rmsprop = lasagne.updates.rmsprop(loss, params, learning_rate=self.learning_rate, epsilon=1e-10)

        # f_train_sgd = theano.function([self.input_var, self.target_var], loss, updates=updates_sgd)
        # f_train_momentum = theano.function([self.input_var, self.target_var], loss, updates=updates_momentum)
        f_train_adam = theano.function([self.input_var, self.target_var], loss, updates=updates_adam)
        # f_train_adagrad = theano.function([self.input_var, self.target_var], loss, updates=updates_adagrad)
        # f_train_adadelta = theano.function([self.input_var, self.target_var], loss, updates=updates_adadelta)
        # f_train_rmsprop = theano.function([self.input_var, self.target_var], loss, updates=updates_rmsprop)

        self.f_train = f_train_adam

    def build_validate_function(self):
        test_preds = lasagne.layers.get_output(self.network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_preds,
                                                                self.target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_preds, axis=1), self.target_var),
                          dtype=theano.config.floatX)

        # Compile a second function computing the validation loss and accuracy:
        self.f_validate = theano.function([self.input_var, self.target_var], [test_loss, test_acc])

    @logging
    def update_learning_rate(self):
        self.learning_rate.set_value(floatX(self.learning_rate.get_value() * 0.1))

VanillaCNNModel.register_class(['vanilla'])


class ResNetTFModel(CIFARModelBase):
    def __init__(self,
                 n=None,
                 train_batch_size=None,
                 validate_batch_size=None):
        super(ResNetTFModel, self).__init__(train_batch_size, validate_batch_size)

        n = n or ParamConfig['n']

        self.learning_rate = theano.shared(lasagne.utils.floatX(ParamConfig['init_learning_rate']))

        # Prepare Theano variables for inputs and targets
        self.input_var = T.tensor4('inputs')
        self.target_var = T.ivector('targets')

        self.network = self.build_cnn(self.input_var, n)
        message("number of parameters in model: %d" % lasagne.layers.count_params(self.network, trainable=True))

        self.saved_init_parameters_values = get_all_param_values(self.network, trainable=True)

        self.build_train_function()
        self.build_validate_function()

    @logging
    def build_cnn(self, input_var=None, n=ParamConfig['n']):
        pass

    @logging
    def build_train_function(self):
        pass

    @logging
    def build_validate_function(self):
        pass

    def get_lr(self, iteration):
        pass

ResNetTFModel.register_class(['tf', 'resnet_tf'])
