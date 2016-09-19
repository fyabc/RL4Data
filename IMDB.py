#! /usr/bin/python

from __future__ import print_function, unicode_literals

import time
import numpy as np
import cPickle as pkl
from collections import OrderedDict

import theano.tensor as T
from theano import shared
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import fX, floatX, logging, get_minibatches_idx, message
from utils_IMDB import prepare_imdb_data as prepare_data, _p, ortho_weight
from config import IMDBConfig

__author__ = 'fyabc'


class IMDBModel(object):
    def __init__(self, reload_model=False):
        self.train_batch_size = IMDBConfig['train_batch_size']
        self.validate_batch_size = IMDBConfig['validate_batch_size']
        self.learning_rate = floatX(IMDBConfig['learning_rate'])

        # Some Theano functions (predictions and updates)
        self.predict_function = None
        self.f_grad_shared = None
        self.f_update = None

        # Parameters of the model (Theano shared variables)
        self.parameters = OrderedDict()

        # Build train function and parameters
        self.use_noise = None
        self.inputs = None
        self.mask = None
        self.targets = None
        self.cost = None

        self.build_train_function()

        if reload_model:
            self.load_model()

    def init_parameters(self):
        np_parameters = OrderedDict()

        rands = np.random.rand(IMDBConfig['n_words'], IMDBConfig['dim_proj'])

        # embedding
        np_parameters['Wemb'] = (0.01 * rands).astype(fX)

        # LSTM
        self.init_lstm_parameters(np_parameters)

        # classifier
        np_parameters['U'] = 0.01 * np.random.randn(IMDBConfig['dim_proj'], IMDBConfig['ydim']).astype(fX)
        np_parameters['b'] = np.zeros((IMDBConfig['ydim'],)).astype(fX)

        for key, value in np_parameters.iteritems():
            self.parameters[key] = shared(value, name=key)

    @staticmethod
    def init_lstm_parameters(np_parameters):
        prefix = 'lstm'
        W = np.concatenate([ortho_weight(IMDBConfig['dim_proj']),
                            ortho_weight(IMDBConfig['dim_proj']),
                            ortho_weight(IMDBConfig['dim_proj']),
                            ortho_weight(IMDBConfig['dim_proj'])], axis=1)
        np_parameters[_p(prefix, 'W')] = W
        U = np.concatenate([ortho_weight(IMDBConfig['dim_proj']),
                            ortho_weight(IMDBConfig['dim_proj']),
                            ortho_weight(IMDBConfig['dim_proj']),
                            ortho_weight(IMDBConfig['dim_proj'])], axis=1)
        np_parameters[_p(prefix, 'U')] = U
        b = np.zeros((4 * IMDBConfig['dim_proj'],))
        np_parameters[_p(prefix, 'b')] = b.astype(fX)

    @logging
    def build_train_function(self):
        # Initialize self.parameters
        self.init_parameters()

        trng = RandomStreams(IMDBConfig['seed'])

        # Build Theano tensor variables.
        self.use_noise = shared(floatX(0.))

    @logging
    def load_model(self, filename=None):
        filename = filename or IMDBConfig['save_to']

    @logging
    def save_model(self, filename=None):
        filename = filename or IMDBConfig['save_to']

    def predict_error(self, data_x, data_y, batch_indices, verbose=False):
        """
        Just compute the error
        f_pred: Theano fct computing the prediction
        prepare_data: usual prepare_data for that dataset.
        """

        valid_err = 0
        for _, valid_index in batch_indices:
            x, mask, y = prepare_data([data_x[t] for t in valid_index],
                                      np.array(data_y)[valid_index],
                                      maxlen=None)
            predicts = self.predict_function(x, mask)
            targets = np.array(data_y)[valid_index]
            valid_err += (predicts == targets).sum()
        valid_err = 1. - floatX(valid_err) / len(data_x)

        return valid_err

    def get_parameter_values(self):
        result = OrderedDict()
        for key in self.parameters:
            result[key] = self.parameters[key].get_value()
        return result

    @logging
    def train(self, train_x, train_y, valid_x, valid_y, test_x, test_y):
        history_errs = []
        best_parameters = {}
        bad_count = 0

        train_size = len(train_x)
        valid_size = len(valid_x)
        test_size = len(test_x)

        kf_valid = get_minibatches_idx(valid_size, self.validate_batch_size)
        kf_test = get_minibatches_idx(test_size, self.validate_batch_size)

        valid_freq = IMDBConfig['valid_freq']
        if valid_freq == -1:
            valid_freq = train_size // self.train_batch_size

        save_freq = IMDBConfig['save_freq']
        if save_freq == -1:
            save_freq = train_size // self.train_batch_size

        display_freq = IMDBConfig['display_freq']
        save_to = IMDBConfig['save_to']
        patience = IMDBConfig['patience']

        update_index = 0  # the number of update done
        early_stop = False  # early stop
        start_time = time.time()

        epoch = 0
        try:
            for epoch in range(IMDBConfig['max_epochs']):
                n_samples = 0

                # Get new shuffled index for the training set.
                kf = get_minibatches_idx(train_size, self.train_batch_size, shuffle=True)

                for _, train_index in kf:
                    update_index += 1

                    # Select the random examples for this minibatch
                    x = [train_x[t] for t in train_index]
                    y = [train_y[t] for t in train_index]

                    # Get the data in numpy.ndarray format
                    # This swap the axis!
                    # Return something of shape (minibatch maxlen, n samples)
                    x, mask, y = prepare_data(x, y)
                    n_samples += x.shape[1]

                    cost = self.f_grad_shared(x, mask, y)
                    self.f_update(self.learning_rate)

                    if np.isnan(cost) or np.isinf(cost):
                        print('bad cost detected: ', cost)
                        return 1., 1., 1.

                    if update_index % display_freq == 0:
                        print('Epoch ', epoch, 'Update ', update_index, 'Cost ', cost)

                    if save_to and update_index % save_freq == 0:
                        print('Saving...')

                        if best_parameters:
                            params = best_parameters
                        else:
                            params = self.get_parameter_values()
                        np.savez(save_to, history_errs=history_errs, **params)
                        pkl.dump(model_options, open('%s.pkl' % save_to, 'wb'))
                        print('Done')

                    if update_index % valid_freq == 0:
                        self.use_noise.set_value(0.)
                        train_err = self.predict_error(train_x, train_y, kf)
                        valid_err = self.predict_error(valid_x, valid_y, kf_valid)
                        test_err = self.predict_error(test_x, test_y, kf_test)

                        history_errs.append([valid_err, test_err])

                        if not best_parameters or valid_err <= np.array(history_errs)[:, 0].min():
                            best_parameters = self.get_parameter_values()
                            bad_counter = 0

                        print(('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err))

                        if len(history_errs) > patience and valid_err >= np.array(history_errs)[:-patience, 0].min():
                            bad_counter += 1
                            if bad_counter > patience:
                                print('Early Stop!')
                                early_stop = True
                                break

                print('Seen %d samples' % n_samples)

                if early_stop:
                    break
        except KeyboardInterrupt:
            print('Training interrupted')

        end_time = time.time()

        kf_train_sorted = get_minibatches_idx(train_size, self.train_batch_size)

        train_err = self.predict_error(train_x, train_y, kf_train_sorted)
        valid_err = self.predict_error(valid_x, valid_y, kf_valid)
        test_err = self.predict_error(test_x, test_y, kf_test)

        print('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err)

        if IMDBConfig['save_to']:
            np.savez(IMDBConfig['save_to'], train_err=train_err,
                     valid_err=valid_err, test_err=test_err,
                     history_errs=history_errs, **best_parameters)

        print('The code run for %d epochs, with %f sec/epochs' % (
            (epoch + 1), (end_time - start_time) / (1. * (epoch + 1))))
        message(('Training took %.1fs' % (end_time - start_time)))
        return train_err, valid_err, test_err
