#! /usr/bin/python

from __future__ import print_function, unicode_literals

import time
import numpy as np

from utils import floatX, logging, get_minibatches_idx, message
from utils import prepare_imdb_data as prepare_data
from config import IMDBConfig

__author__ = 'fyabc'


class IMDBModel(object):
    def __init__(self, reload_model=False):
        self.train_batch_size = IMDBConfig['train_batch_size']
        self.validate_batch_size = IMDBConfig['validate_batch_size']

        # Some Theano functions (predictions and updates)
        self.predict_function = None
        self.f_grad_shared = None
        self.f_update = None

        # Parameters of the model (Theano shared variables)
        self.parameters = None

        if reload_model:
            self.load_model()

    @logging
    def load_model(self, filename=None):
        filename = filename or IMDBConfig['save_to']

    @logging
    def save_model(self, filename=None):
        filename = filename or IMDBConfig['save_to']

    def pred_error(self, data_x, data_y, batch_indices, verbose=False):
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

        update_index = 0  # the number of update done
        estop = False  # early stop
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
                    x = [train_x[t]for t in train_index]
                    y = [train_y[t] for t in train_index]

                    # Get the data in numpy.ndarray format
                    # This swap the axis!
                    # Return something of shape (minibatch maxlen, n samples)
                    x, mask, y = prepare_data(x, y)
                    n_samples += x.shape[1]

                    # TODO
                pass
        except KeyboardInterrupt:
            print('Training interrupted')

        end_time = time.time()

        kf_train_sorted = get_minibatches_idx(train_size, self.train_batch_size)

        train_err = self.pred_error(train_x, train_y, kf_train_sorted)
        valid_err = self.pred_error(valid_x, valid_y, kf_valid)
        test_err = self.pred_error(test_x, test_y, kf_test)

        print('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err)

        if IMDBConfig['save_to']:
            np.savez(IMDBConfig['save_to'], train_err=train_err,
                     valid_err=valid_err, test_err=test_err,
                     history_errs=history_errs, **best_parameters)

        print('The code run for %d epochs, with %f sec/epochs' % (
            (epoch + 1), (end_time - start_time) / (1. * (epoch + 1))))
        message(('Training took %.1fs' % (end_time - start_time)))
        return train_err, valid_err, test_err
