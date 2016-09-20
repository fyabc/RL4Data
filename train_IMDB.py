# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import time
import numpy as np

from config import IMDBConfig
from IMDB import IMDBModel
from utils import process_before_train, floatX, message
from utils_IMDB import load_imdb_data, preprocess_imdb_data, get_minibatches_idx
from utils_IMDB import prepare_imdb_data as prepare_data

from policyNetwork import PolicyNetwork

__author__ = 'fyabc'


def train_IMDB():
    np.random.seed(IMDBConfig['seed'])

    # Loading data
    train_data, valid_data, test_data = load_imdb_data(n_words=IMDBConfig['n_words'])
    train_data, valid_data, test_data = preprocess_imdb_data(train_data, valid_data, test_data)

    train_x, train_y = train_data
    valid_x, valid_y = valid_data
    test_x, test_y = test_data
    
    train_size = len(train_x)
    valid_size = len(valid_x)
    test_size = len(test_x)

    print("%d train examples" % train_size)
    print("%d valid examples" % valid_size)
    print("%d test examples" % test_size)

    # Build model
    imdb = IMDBModel(IMDBConfig['reload_model'])

    # Build policy
    policy = PolicyNetwork(input_size=4)

    # Training
    history_errs = []
    best_parameters = {}
    bad_counter = 0
    
    kf_valid = get_minibatches_idx(valid_size, imdb.validate_batch_size)
    kf_test = get_minibatches_idx(test_size, imdb.validate_batch_size)

    valid_freq = IMDBConfig['valid_freq']
    if valid_freq == -1:
        valid_freq = train_size // imdb.train_batch_size

    save_freq = IMDBConfig['save_freq']
    if save_freq == -1:
        save_freq = train_size // imdb.train_batch_size

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
            kf = get_minibatches_idx(train_size, imdb.train_batch_size, shuffle=True)

            for _, train_index in kf:
                update_index += 1
                imdb.use_noise.set_value(floatX(1.))

                # Select the random examples for this minibatch
                x = [train_x[t] for t in train_index]
                y = [train_y[t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)

                # TODO
                # Policy take action here

                probabilities = imdb.f_predict_prob(x, mask)
                print(probabilities)

                n_samples += x.shape[1]

                cost = imdb.f_grad_shared(x, mask, y)
                imdb.f_update(imdb.learning_rate)

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
                        params = imdb.get_parameter_values()
                    np.savez(save_to, history_errs=history_errs, **params)
                    # pkl.dump(IMDBConfig, open('%s.pkl' % save_to, 'wb'))
                    print('Done')

                if update_index % valid_freq == 0:
                    imdb.use_noise.set_value(0.)
                    train_err = imdb.predict_error(train_x, train_y, kf)
                    valid_err = imdb.predict_error(valid_x, valid_y, kf_valid)
                    test_err = imdb.predict_error(test_x, test_y, kf_test)

                    history_errs.append([valid_err, test_err])

                    if not best_parameters or valid_err <= np.array(history_errs)[:, 0].min():
                        best_parameters = imdb.get_parameter_values()
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

    kf_train_sorted = get_minibatches_idx(train_size, imdb.train_batch_size)

    train_err = imdb.predict_error(train_x, train_y, kf_train_sorted)
    valid_err = imdb.predict_error(valid_x, valid_y, kf_valid)
    test_err = imdb.predict_error(test_x, test_y, kf_test)

    print('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err)

    if IMDBConfig['save_to']:
        np.savez(IMDBConfig['save_to'], train_err=train_err,
                 valid_err=valid_err, test_err=test_err,
                 history_errs=history_errs, **best_parameters)

    print('The code run for %d epochs, with %f sec/epochs' % (
        (epoch + 1), (end_time - start_time) / (1. * (epoch + 1))))
    message(('Training took %.1fs' % (end_time - start_time)))
    return train_err, valid_err, test_err
    

if __name__ == '__main__':
    process_before_train(IMDBConfig)

    train_IMDB()
