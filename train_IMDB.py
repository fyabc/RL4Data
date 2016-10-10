# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import time
import numpy as np

from config import IMDBConfig, Config, PolicyConfig
from IMDB import IMDBModel
from utils import process_before_train, floatX, message, get_part_data
from utils_IMDB import load_imdb_data, preprocess_imdb_data, get_minibatches_idx
from utils_IMDB import prepare_imdb_data as prepare_data

from policyNetwork import PolicyNetwork

__author__ = 'fyabc'


def pre_process_data():
    np.random.seed(IMDBConfig['seed'])

    # Loading data
    train_data, valid_data, test_data = load_imdb_data(n_words=IMDBConfig['n_words'],
                                                       valid_portion=IMDBConfig['valid_portion'],
                                                       maxlen=IMDBConfig['maxlen'])
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

    return train_x, train_y, valid_x, valid_y, test_x, test_y, \
           train_size, valid_size, test_size


def pre_process_config(imdb, train_size, valid_size, test_size):
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

    return kf_valid, kf_test, valid_freq, save_freq, display_freq, save_to, patience


def save_parameters(imdb, best_parameters, save_to, history_errs):
    print('Saving...')

    if best_parameters:
        params = best_parameters
    else:
        params = imdb.get_parameter_values()
    np.savez(save_to, history_errs=history_errs, **params)
    # pkl.dump(IMDBConfig, open('%s.pkl' % save_to, 'wb'))
    print('Done')


def test_and_post_process(imdb,
                          train_size, train_x, train_y, valid_x, valid_y, test_x, test_y,
                          kf_valid, kf_test,
                          history_errs, best_parameters,
                          epoch, start_time, end_time,
                          save_to):
    kf_train_sorted = get_minibatches_idx(train_size, imdb.train_batch_size)

    train_err = imdb.predict_error(train_x, train_y, kf_train_sorted)
    valid_err = imdb.predict_error(valid_x, valid_y, kf_valid)
    test_err = imdb.predict_error(test_x, test_y, kf_test)

    print('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err)

    if save_to:
        np.savez(save_to, train_err=train_err,
                 valid_err=valid_err, test_err=test_err,
                 history_errs=history_errs, **best_parameters)

    print('The code run for %d epochs, with %f sec/epochs' % (
        (epoch + 1), (end_time - start_time) / (1. * (epoch + 1))))
    message(('Training took %.1fs' % (end_time - start_time)))
    return train_err, valid_err, test_err


def train_raw_IMDB():
    np.random.seed(IMDBConfig['seed'])

    # Loading data
    train_x, train_y, valid_x, valid_y, test_x, test_y, \
    train_size, valid_size, test_size = pre_process_data()

    # Building model
    imdb = IMDBModel(IMDBConfig['reload_model'])

    # Loading configure settings
    kf_valid, kf_test, \
    valid_freq, save_freq, display_freq, \
    save_to, patience = pre_process_config(imdb, train_size, valid_size, test_size)

    # Training
    history_errs = []
    best_parameters = None
    bad_counter = 0

    update_index = 0  # the number of update done
    early_stop = False  # early stop
    start_time = time.time()

    epoch = 0
    history_train_costs = []

    try:
        total_n_samples = 0

        for epoch in range(IMDBConfig['epoch_per_episode']):
            print('[Epoch {}]'.format(epoch))
            message('[Epoch {}]'.format(epoch))

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
                x, mask, y = prepare_data(x, np.asarray(y, dtype='int64'))

                n_samples += x.shape[1]
                total_n_samples += x.shape[1]

                cost = imdb.f_grad_shared(x, mask, y)
                imdb.f_update(imdb.learning_rate)

                history_train_costs.append(cost)

                if np.isnan(cost) or np.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if update_index % display_freq == 0:
                    print('Epoch ', epoch, 'Update ', update_index, 'Cost ', cost)

                if save_to and update_index % save_freq == 0:
                    save_parameters(imdb, best_parameters, save_to, history_errs)

                if update_index % valid_freq == 0:
                    imdb.use_noise.set_value(0.)
                    # train_err = imdb.predict_error(train_x, train_y, kf)
                    valid_err = imdb.predict_error(valid_x, valid_y, kf_valid)
                    test_err = imdb.predict_error(test_x, test_y, kf_test)

                    history_errs.append([valid_err, test_err])

                    if best_parameters is None or valid_err <= np.array(history_errs)[:, 0].min():
                        best_parameters = imdb.get_parameter_values()
                        bad_counter = 0

                    print('Train', 0.00, 'Valid', valid_err, 'Test', test_err,
                          'Total_samples', total_n_samples)

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

    test_and_post_process(imdb,
                          train_size, train_x, train_y, valid_x, valid_y, test_x, test_y,
                          kf_valid, kf_test,
                          history_errs, best_parameters,
                          epoch, start_time, end_time,
                          save_to)


def train_policy_IMDB():
    np.random.seed(IMDBConfig['seed'])

    # Loading data
    train_x, train_y, valid_x, valid_y, test_x, test_y, \
        train_size, valid_size, test_size = pre_process_data()

    # Building model
    imdb = IMDBModel(IMDBConfig['reload_model'])

    # Loading configure settings
    kf_valid, kf_test, \
        valid_freq, save_freq, display_freq, \
        save_to, patience = pre_process_config(imdb, train_size, valid_size, test_size)

    # Build policy
    input_size = imdb.get_policy_input_size()
    print('Input size of policy network:', input_size)
    policy = PolicyNetwork(input_size=input_size, start_b=2.)

    num_episodes = PolicyConfig['num_episodes']

    for episode in range(num_episodes):
        print('[Episode {}]'.format(episode))
        message('[Episode {}]'.format(episode))

        imdb.reset_parameters()

        # get small training data
        train_small_size = IMDBConfig['train_small_size']
        train_small_x, train_small_y = get_part_data(np.asarray(train_x), np.asarray(train_y), train_small_size)

        # Training
        history_errs = []
        best_parameters = None
        bad_counter = 0

        update_index = 0  # the number of update done
        early_stop = False  # early stop
        start_time = time.time()

        epoch = 0
        history_train_costs = []

        try:
            total_n_samples = 0

            for epoch in range(IMDBConfig['epoch_per_episode']):
                print('[Epoch {}]'.format(epoch))
                message('[Epoch {}]'.format(epoch))

                n_samples = 0

                # Get new shuffled index for the training set.
                kf = get_minibatches_idx(train_small_size, imdb.train_batch_size, shuffle=True)

                policy.start_new_epoch()

                for _, train_index in kf:
                    update_index += 1
                    imdb.use_noise.set_value(floatX(1.))

                    # Select the random examples for this minibatch
                    x = [train_small_x[t] for t in train_index]
                    y = [train_small_y[t] for t in train_index]

                    # Get the data in numpy.ndarray format
                    # This swap the axis!
                    # Return something of shape (minibatch maxlen, n samples)
                    x, mask, y = prepare_data(x, np.asarray(y, dtype='int64'))

                    # Policy take action here
                    probability = imdb.get_policy_input(x, mask, y, epoch)
                    actions = policy.take_action(probability)

                    # get masked inputs and targets
                    x = x[:, actions]
                    mask = mask[:, actions]
                    y = y[actions]

                    n_samples += x.shape[1]
                    total_n_samples += x.shape[1]

                    cost = imdb.f_grad_shared(x, mask, y)
                    imdb.f_update(imdb.learning_rate)

                    history_train_costs.append(cost)

                    if np.isnan(cost) or np.isinf(cost):
                        print('bad cost detected: ', cost)
                        return 1., 1., 1.

                    if update_index % display_freq == 0:
                        print('Epoch ', epoch, 'Update ', update_index, 'Cost ', cost)

                    # Do not save when training policy!

                    if update_index % valid_freq == 0:
                        imdb.use_noise.set_value(0.)
                        # train_err = imdb.predict_error(train_x, train_y, kf)
                        valid_err = imdb.predict_error(valid_x, valid_y, kf_valid)
                        test_err = imdb.predict_error(test_x, test_y, kf_test)

                        history_errs.append([valid_err, test_err])

                        if best_parameters is None or valid_err <= np.array(history_errs)[:, 0].min():
                            best_parameters = imdb.get_parameter_values()
                            bad_counter = 0

                        print('Train', 0.00, 'Valid', valid_err, 'Test', test_err,
                              'Total_samples', total_n_samples)

                        if len(history_errs) > patience and valid_err >= np.array(history_errs)[:-patience, 0].min():
                            bad_counter += 1
                            if bad_counter > patience:
                                print('Early Stop!')
                                early_stop = True
                                break

                print('Seen %d samples' % n_samples)

                # Immediate reward
                if PolicyConfig['immediate_reward']:
                    imdb.use_noise.set_value(0.)
                    valid_err = imdb.predict_error(valid_x, valid_y, kf_valid)
                    policy.reward_buffer.append(1. - valid_err)

                if early_stop:
                    break
        except KeyboardInterrupt:
            print('Training interrupted')

        end_time = time.time()

        train_err, valid_err, test_err = test_and_post_process(
            imdb,
            train_small_size, train_small_x, train_small_y, valid_x, valid_y, test_x,
            test_y,
            kf_valid, kf_test,
            history_errs, best_parameters,
            epoch, start_time, end_time,
            save_to=False)

        # Updating policy
        policy.update(1. - valid_err)

        if Config['policy_save_freq'] > 0 and episode % Config['policy_save_freq'] == 0:
            policy.save_policy()

        if episode % PolicyConfig['policy_learning_rate_discount_freq'] == 0:
            policy.discount_learning_rate()


def train_deterministic_stochastic_IMDB():
    np.random.seed(IMDBConfig['seed'])

    # Loading data
    train_x, train_y, valid_x, valid_y, test_x, test_y, \
    train_size, valid_size, test_size = pre_process_data()

    # Building model
    imdb = IMDBModel(IMDBConfig['reload_model'])

    # Loading configure settings
    kf_valid, kf_test, \
        valid_freq, save_freq, display_freq, \
        save_to, patience = pre_process_config(imdb, train_size, valid_size, test_size)

    # Build policy
    input_size = imdb.get_policy_input_size()
    print('Input size of policy network:', input_size)
    policy = PolicyNetwork(input_size=input_size, start_b=0.)
    policy.load_policy()
    message('$    w = {}\n'
            '$    b = {}'
            .format(policy.W.get_value(), policy.b.get_value()))

    # Training
    history_errs = []
    best_parameters = None
    bad_counter = 0

    update_index = 0  # the number of update done
    early_stop = False  # early stop
    start_time = time.time()

    epoch = 0
    history_train_costs = []

    try:
        total_n_samples = 0

        for epoch in range(1, 1 + IMDBConfig['epoch_per_episode']):
            print('[Epoch {}]'.format(epoch))
            message('[Epoch {}]'.format(epoch))

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
                x, mask, y = prepare_data(x, np.asarray(y, dtype='int64'))

                # Policy take action here
                probability = imdb.get_policy_input(x, mask, y, epoch)
                if Config['train_type'] == 'stochastic':
                    actions = policy.take_action(probability, False)

                    # get masked inputs and targets
                    x = x[:, actions]
                    mask = mask[:, actions]
                    y = y[actions]

                    n_samples += x.shape[1]
                    total_n_samples += x.shape[1]

                    cost = imdb.f_grad_shared(x, mask, y)
                    imdb.f_update(imdb.learning_rate)
                elif Config['train_type'] == 'deterministic':
                    raise NotImplementedError('Deterministic is not implemented')

                history_train_costs.append(cost)

                if np.isnan(cost) or np.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if update_index % display_freq == 0:
                    print('Epoch ', epoch, 'Update ', update_index, 'Cost ', cost)

                if save_to and update_index % save_freq == 0:
                    save_parameters(imdb, best_parameters, save_to, history_errs)

                if update_index % valid_freq == 0:
                    imdb.use_noise.set_value(0.)
                    # train_err = imdb.predict_error(train_x, train_y, kf)
                    valid_err = imdb.predict_error(valid_x, valid_y, kf_valid)
                    test_err = imdb.predict_error(test_x, test_y, kf_test)

                    history_errs.append([valid_err, test_err])

                    if best_parameters is None or valid_err <= np.array(history_errs)[:, 0].min():
                        best_parameters = imdb.get_parameter_values()
                        bad_counter = 0

                    print('Train', 0.0, 'Valid', valid_err, 'Test', test_err,
                          'Total_samples', total_n_samples)

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

    test_and_post_process(imdb,
                          train_size, train_x, train_y, valid_x, valid_y, test_x, test_y,
                          kf_valid, kf_test,
                          history_errs, best_parameters,
                          epoch, start_time, end_time,
                          False)


if __name__ == '__main__':
    process_before_train(IMDBConfig)

    if Config['train_type'] == 'raw':
        train_raw_IMDB()
    elif Config['train_type'] == 'policy':
        train_policy_IMDB()
    elif Config['train_type'] == 'deterministic':
        train_deterministic_stochastic_IMDB()
    elif Config['train_type'] == 'stochastic':
        train_deterministic_stochastic_IMDB()
    else:
        raise Exception('Unknown train type {}'.format(Config['train_type']))
