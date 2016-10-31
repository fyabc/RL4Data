# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys
import traceback
import time
import numpy as np
import heapq
from collections import deque

from config import IMDBConfig as ParamConfig, Config, PolicyConfig
from model_IMDB import IMDBModel
from utils import process_before_train, floatX, message, get_part_data, finalize_logging_file, get_minibatches_idx
from utils_IMDB import load_imdb_data, preprocess_imdb_data
from utils_IMDB import prepare_imdb_data as prepare_data

# Actor-Critic from ChangXu
from criticNetwork import CriticNetwork

from policyNetwork import PolicyNetwork

__author__ = 'fyabc'


def pre_process_data():
    np.random.seed(ParamConfig['seed'])

    # Loading data
    train_data, valid_data, test_data = load_imdb_data(n_words=ParamConfig['n_words'],
                                                       valid_portion=ParamConfig['valid_portion'],
                                                       maxlen=ParamConfig['maxlen'])
    train_data, valid_data, test_data = preprocess_imdb_data(train_data, valid_data, test_data)

    train_x, train_y = train_data
    valid_x, valid_y = valid_data
    test_x, test_y = test_data

    train_size = len(train_x)
    valid_size = len(valid_x)
    test_size = len(test_x)

    message("%d train examples" % train_size)
    message("%d valid examples" % valid_size)
    message("%d test examples" % test_size)

    return train_x, train_y, valid_x, valid_y, test_x, test_y, \
        train_size, valid_size, test_size


def pre_process_config(model, train_size, valid_size, test_size):
    kf_valid = get_minibatches_idx(valid_size, model.validate_batch_size)
    kf_test = get_minibatches_idx(test_size, model.validate_batch_size)

    valid_freq = ParamConfig['valid_freq']
    if valid_freq == -1:
        valid_freq = train_size // model.train_batch_size

    save_freq = ParamConfig['save_freq']
    if save_freq == -1:
        save_freq = train_size // model.train_batch_size

    display_freq = ParamConfig['display_freq']
    save_to = ParamConfig['save_to']
    patience = ParamConfig['patience']

    return kf_valid, kf_test, valid_freq, save_freq, display_freq, save_to, patience


def save_parameters(model, best_parameters, save_to, history_errs):
    message('Saving...')

    if best_parameters:
        params = best_parameters
    else:
        params = model.get_parameter_values()
    np.savez(save_to, history_errs=history_errs, **params)
    # pkl.dump(IMDBConfig, open('%s.pkl' % save_to, 'wb'))
    message('Done')


def test_and_post_process(model,
                          train_size, train_x, train_y, valid_x, valid_y, test_x, test_y,
                          kf_valid, kf_test,
                          history_errs, best_parameters,
                          epoch, start_time, end_time,
                          save_to):
    kf_train_sorted = get_minibatches_idx(train_size, model.train_batch_size)

    train_err = model.predict_error(train_x, train_y, kf_train_sorted)
    valid_err = model.predict_error(valid_x, valid_y, kf_valid)
    test_err = model.predict_error(test_x, test_y, kf_test)

    message('Final: Train ', train_err, 'Valid ', valid_err, 'Test ', test_err)

    if save_to:
        np.savez(save_to, train_err=train_err,
                 valid_err=valid_err, test_err=test_err,
                 history_errs=history_errs, **best_parameters)

    message('The code run for %d epochs, with %f sec/epochs' % (
        (epoch + 1), (end_time - start_time) / (1. * (epoch + 1))))
    message(('Training took %.1fs' % (end_time - start_time)))
    return train_err, valid_err, test_err


def train_raw_IMDB():
    np.random.seed(ParamConfig['seed'])

    # Loading data
    train_x, train_y, valid_x, valid_y, test_x, test_y, \
        train_size, valid_size, test_size = pre_process_data()

    # Building model
    model = IMDBModel(ParamConfig['reload_model'])

    # Loading configure settings
    kf_valid, kf_test, \
        valid_freq, save_freq, display_freq, \
        save_to, patience = pre_process_config(model, train_size, valid_size, test_size)

    # Training
    history_errs = []
    best_parameters = None
    bad_counter = 0
    update_index = 0  # the number of update done
    early_stop = False  # early stop
    epoch = 0
    history_train_costs = []

    start_time = time.time()

    try:
        total_n_samples = 0

        for epoch in range(ParamConfig['epoch_per_episode']):
            print('[Epoch {}]'.format(epoch))
            message('[Epoch {}]'.format(epoch))

            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(train_size, model.train_batch_size, shuffle=True)

            for _, train_index in kf:
                update_index += 1
                model.use_noise.set_value(floatX(1.))

                # Select the random examples for this minibatch
                x = [train_x[t] for t in train_index]
                y = [train_y[t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, np.asarray(y, dtype='int64'))

                n_samples += x.shape[1]
                total_n_samples += x.shape[1]

                cost = model.f_train(x, mask, y)

                history_train_costs.append(cost)

                if cost is not None and (np.isnan(cost) or np.isinf(cost)):
                    message('bad cost detected: ', cost)
                    return 1., 1., 1.

                if update_index % display_freq == 0:
                    message('Epoch ', epoch, 'Update ', update_index, 'Cost ', cost)

                if save_to and update_index % save_freq == 0:
                    save_parameters(model, best_parameters, save_to, history_errs)

                if update_index % ParamConfig['train_loss_freq'] == 0:
                    train_loss = model.get_training_loss(train_x, train_y)
                    message('Training Loss:', train_loss)

                if update_index % valid_freq == 0:
                    model.use_noise.set_value(0.)
                    valid_err = model.predict_error(valid_x, valid_y, kf_valid)
                    test_err = model.predict_error(test_x, test_y, kf_test)

                    history_errs.append([valid_err, test_err])

                    if best_parameters is None or valid_err <= np.array(history_errs)[:, 0].min():
                        best_parameters = model.get_parameter_values()
                        bad_counter = 0

                    message('Train', 0.00, 'Valid', valid_err, 'Test', test_err,
                            'Total_samples', total_n_samples)

                    if len(history_errs) > patience and valid_err >= np.array(history_errs)[:-patience, 0].min():
                        bad_counter += 1
                        if bad_counter > patience:
                            message('Early Stop!')
                            early_stop = True
                            break

            message('Seen %d samples' % n_samples)

            if early_stop:
                break
    except KeyboardInterrupt:
        message('Training interrupted')

    end_time = time.time()

    test_and_post_process(model,
                          train_size, train_x, train_y, valid_x, valid_y, test_x, test_y,
                          kf_valid, kf_test,
                          history_errs, best_parameters,
                          epoch, start_time, end_time,
                          save_to)


def train_SPL_IMDB():
    np.random.seed(ParamConfig['seed'])

    # Loading data
    train_x, train_y, valid_x, valid_y, test_x, test_y, \
        train_size, valid_size, test_size = pre_process_data()

    # Building model
    model = IMDBModel(ParamConfig['reload_model'])

    # Loading configure settings
    kf_valid, kf_test, \
        valid_freq, save_freq, display_freq, \
        save_to, patience = pre_process_config(model, train_size, valid_size, test_size)

    # # Self-paced learning iterate on data cases
    total_iteration_number = ParamConfig['epoch_per_episode'] * train_size // model.train_batch_size

    # Get the cost threshold \lambda.
    def cost_threshold(iteration):
        return 1 + (model.train_batch_size - 1) * iteration / total_iteration_number

    # Data buffer
    spl_buffer = deque()

    # When collect such number of cases, update them.
    update_maxlen = model.train_batch_size

    # # Self-paced learning setting end

    # Training
    history_errs = []
    best_parameters = None
    bad_counter = 0
    iteration = 0  # the number of update done
    early_stop = False  # early stop
    epoch = 0
    history_train_costs = []

    start_time = time.time()

    try:
        total_n_samples = 0

        for epoch in range(ParamConfig['epoch_per_episode']):
            print('[Epoch {}]'.format(epoch))
            message('[Epoch {}]'.format(epoch))
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(train_size, model.train_batch_size, shuffle=True)

            for _, train_index in kf:
                iteration += 1
                model.use_noise.set_value(floatX(1.))

                # Select the random examples for this minibatch
                x = [train_x[t] for t in train_index]
                y = [train_y[t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, np.asarray(y, dtype='int64'))

                # Self-paced learning check
                selected_number = cost_threshold(iteration)

                cost_list = model.f_cost_list_without_decay(x, mask, y)
                positive_cost_list = cost_list[y == 1]
                negative_cost_list = cost_list[y == 0]

                if positive_cost_list.size != 0:
                    positive_threshold = heapq.nsmallest(selected_number, positive_cost_list)[-1]
                    for i in range(len(y)):
                        if y[i] == 1 and cost_list[i] <= positive_threshold:
                            spl_buffer.append(train_index[i])
                if negative_cost_list.size != 0:
                    negative_threshold = heapq.nsmallest(selected_number, negative_cost_list)[-1]
                    for i in range(len(y)):
                        if y[i] == 0 and cost_list[i] <= negative_threshold:
                            spl_buffer.append(train_index[i])

                if len(spl_buffer) >= update_maxlen:
                    # message('SPL buffer full, update...', end='')

                    update_batch_index = [spl_buffer.popleft() for _ in range(update_maxlen)]
                    x_selected = [train_x[t] for t in update_batch_index]
                    y_selected = [train_y[t] for t in update_batch_index]
                    x_selected, mask_selected, y_selected = prepare_data(
                        x_selected, np.asarray(y_selected, dtype='int64'))

                    n_samples += x_selected.shape[1]
                    total_n_samples += x_selected.shape[1]

                    cost = model.f_train(x_selected, mask_selected, y_selected)

                    # message('done')
                else:
                    cost = None

                history_train_costs.append(cost)

                if cost is not None and (np.isnan(cost) or np.isinf(cost)):
                    message('bad cost detected: ', cost)
                    return 1., 1., 1.

                # # In SPL, display after each update
                # if iteration % display_freq == 0:
                if cost is not None:
                    message('Epoch ', epoch, 'Update ', iteration, 'Cost ', cost)

                # Do not save when running SPL!

                if iteration % ParamConfig['train_loss_freq'] == 0:
                    train_loss = model.get_training_loss(train_x, train_y)
                    message('Training Loss:', train_loss)

                if iteration % valid_freq == 0:
                    model.use_noise.set_value(0.)
                    valid_err = model.predict_error(valid_x, valid_y, kf_valid)
                    test_err = model.predict_error(test_x, test_y, kf_test)

                    history_errs.append([valid_err, test_err])

                    if best_parameters is None or valid_err <= np.array(history_errs)[:, 0].min():
                        best_parameters = model.get_parameter_values()
                        bad_counter = 0

                    message('Train', 0.00, 'Valid', valid_err, 'Test', test_err,
                            'Total_samples', total_n_samples)

                    if len(history_errs) > patience and valid_err >= np.array(history_errs)[:-patience, 0].min():
                        bad_counter += 1
                        if bad_counter > patience:
                            message('Early Stop!')
                            early_stop = True
                            break

            message('Seen %d samples' % n_samples)

            if early_stop:
                break
    except KeyboardInterrupt:
        message('Training interrupted')

    end_time = time.time()

    test_and_post_process(model,
                          train_size, train_x, train_y, valid_x, valid_y, test_x, test_y,
                          kf_valid, kf_test,
                          history_errs, best_parameters,
                          epoch, start_time, end_time,
                          save_to)


def train_policy_IMDB():
    np.random.seed(ParamConfig['seed'])

    # Loading data
    train_x, train_y, valid_x, valid_y, test_x, test_y, \
        train_size, valid_size, test_size = pre_process_data()

    # Building model
    model = IMDBModel(ParamConfig['reload_model'])

    # Loading configure settings
    kf_valid, kf_test, \
        valid_freq, save_freq, display_freq, \
        save_to, patience = pre_process_config(model, train_size, valid_size, test_size)

    # Build policy
    input_size = model.get_policy_input_size()
    print('Input size of policy network:', input_size)
    policy = PolicyNetwork(input_size=input_size, start_b=PolicyConfig['b_init'])

    # Temp code.
    if Config['temp_job']:
        policy.load_policy(filename='./data/imdb/Pi_terminal_policy_speed_trained.npz')
        policy.b.set_value(floatX(-1.0))

    for episode in range(PolicyConfig['num_episodes']):
        print('[Episode {}]'.format(episode))
        message('[Episode {}]'.format(episode))

        model.reset_parameters()

        # get small training data
        train_small_x, train_small_y = get_part_data(np.asarray(train_x), np.asarray(train_y),
                                                     ParamConfig['train_small_size'])
        train_small_size = len(train_small_x)

        # Training
        history_errs = []
        best_parameters = None
        bad_counter = 0
        iteration = 0  # the number of update done
        early_stop = False  # early stop
        epoch = 0
        history_accuracy = []

        # Speed reward
        first_over_iteration = None

        start_time = time.time()

        try:
            total_n_samples = 0

            for epoch in range(ParamConfig['epoch_per_episode']):
                print('[Epoch {}]'.format(epoch))
                message('[Epoch {}]'.format(epoch))

                n_samples = 0

                # Get new shuffled index for the training set.
                kf = get_minibatches_idx(train_small_size, model.train_batch_size, shuffle=True)

                policy.start_new_epoch()

                for _, train_index in kf:
                    iteration += 1
                    model.use_noise.set_value(floatX(1.))

                    # Select the random examples for this minibatch
                    x = [train_small_x[t] for t in train_index]
                    y = [train_small_y[t] for t in train_index]

                    # Get the data in numpy.ndarray format
                    # This swap the axis!
                    # Return something of shape (minibatch maxlen, n samples)
                    x, mask, y = prepare_data(x, np.asarray(y, dtype='int64'))

                    # Policy take action here
                    probability = model.get_policy_input(x, mask, y, epoch, history_accuracy)

                    actions = policy.take_action(probability)

                    # get masked inputs and targets
                    x = x[:, actions]
                    mask = mask[:, actions]
                    y = y[actions]

                    n_samples += x.shape[1]
                    total_n_samples += x.shape[1]

                    cost = model.f_train(x, mask, y)

                    if cost is not None and (np.isnan(cost) or np.isinf(cost)):
                        message('bad cost detected: ', cost)
                        return 1., 1., 1.

                    if iteration % display_freq == 0:
                        message('Epoch ', epoch, 'Update ', iteration, 'Cost ', cost)

                    # Do not save when training policy!

                    if iteration % ParamConfig['train_loss_freq'] == 0:
                        train_loss = model.get_training_loss(train_x, train_y)
                        message('Training Loss:', train_loss)

                    if iteration % valid_freq == 0:
                        model.use_noise.set_value(0.)
                        # train_err = model.predict_error(train_x, train_y, kf)
                        valid_err = model.predict_error(valid_x, valid_y, kf_valid)
                        test_err = model.predict_error(test_x, test_y, kf_test)

                        history_errs.append([valid_err, test_err])
                        history_accuracy.append(1. - valid_err)

                        # Check speed rewards
                        if first_over_iteration is None and 1. - valid_err >= PolicyConfig['speed_reward_threshold']:
                            first_over_iteration = iteration

                        if best_parameters is None or valid_err <= np.array(history_errs)[:, 0].min():
                            best_parameters = model.get_parameter_values()
                            bad_counter = 0

                        message('Train', 0.00, 'Valid', valid_err, 'Test', test_err,
                                'Total_samples', total_n_samples)

                        if len(history_errs) > patience and valid_err >= np.array(history_errs)[:-patience, 0].min():
                            bad_counter += 1
                            if bad_counter > patience:
                                message('Early Stop!')
                                early_stop = True
                                break

                message('Seen %d samples' % n_samples)

                # Immediate reward
                if PolicyConfig['immediate_reward']:
                    model.use_noise.set_value(0.)
                    valid_err = model.predict_error(valid_x, valid_y, kf_valid)
                    policy.reward_buffer.append(1. - valid_err)

                if early_stop:
                    break
        except KeyboardInterrupt:
            message('Training interrupted')

        end_time = time.time()

        train_err, valid_err, test_err = test_and_post_process(
            model,
            train_small_size, train_small_x, train_small_y, valid_x, valid_y, test_x,
            test_y,
            kf_valid, kf_test,
            history_errs, best_parameters,
            epoch, start_time, end_time,
            save_to=False)

        # Updating policy
        if PolicyConfig['speed_reward']:
            if first_over_iteration is None:
                first_over_iteration = iteration + 1
            terminal_reward = float(first_over_iteration) / iteration
            policy.update(-np.log(terminal_reward))

            message('First over index:', first_over_iteration)
            message('Total index:', iteration)
            message('Terminal reward:', terminal_reward)
        else:
            policy.update(1. - valid_err)

        if Config['policy_save_freq'] > 0 and episode % Config['policy_save_freq'] == 0:
            policy.save_policy()


def train_actor_critic_IMDB():
    np.random.seed(ParamConfig['seed'])

    # Loading data
    train_x, train_y, valid_x, valid_y, test_x, test_y, \
        train_size, valid_size, test_size = pre_process_data()

    # Building model
    model = IMDBModel(ParamConfig['reload_model'])

    # Loading configure settings
    kf_valid, kf_test, \
        valid_freq, save_freq, display_freq, \
        save_to, patience = pre_process_config(model, train_size, valid_size, test_size)

    kf_valid_part = get_minibatches_idx(PolicyConfig['immediate_reward_sample_size'], model.validate_batch_size)

    # Build Actor and Critic network
    input_size = model.get_policy_input_size()
    print('Input size of policy network:', input_size)
    actor = PolicyNetwork(input_size=input_size, start_b=PolicyConfig['b_init'])
    critic = CriticNetwork(feature_size=input_size, batch_size=model.train_batch_size)

    num_episodes = PolicyConfig['num_episodes']

    for episode in range(num_episodes):
        print('[Episode {}]'.format(episode))
        message('[Episode {}]'.format(episode))

        model.reset_parameters()

        # get small training data
        train_small_x, train_small_y = get_part_data(np.asarray(train_x), np.asarray(train_y),
                                                     ParamConfig['train_small_size'])
        train_small_size = len(train_small_x)

        # Training
        history_errs = []
        best_parameters = None
        bad_counter = 0
        iteration = 0  # the number of update done
        early_stop = False  # early stop
        epoch = 0
        history_accuracy = []

        start_time = time.time()

        try:
            total_n_samples = 0

            for epoch in range(ParamConfig['epoch_per_episode']):
                print('[Epoch {}]'.format(epoch))
                message('[Epoch {}]'.format(epoch))

                n_samples = 0

                # Get new shuffled index for the training set.
                kf = get_minibatches_idx(train_small_size, model.train_batch_size, shuffle=True)

                for cur_batch_idx, train_index in kf:
                    if len(train_index) < model.train_batch_size:
                        continue

                    iteration += 1

                    # IMDB set noise before each batch
                    model.use_noise.set_value(floatX(1.))

                    x = [train_small_x[t] for t in train_index]
                    y = [train_small_y[t] for t in train_index]

                    # Get the data in numpy.ndarray format
                    # This swap the axis!
                    # Return something of shape (minibatch maxlen, n samples)
                    x, mask, y = prepare_data(x, np.asarray(y, dtype='int64'))

                    probability = model.get_policy_input(x, mask, y, epoch, history_accuracy)
                    actions = actor.take_action(probability, log_replay=False)

                    # get masked inputs and targets
                    x_selected = x[:, actions]
                    mask_selected = mask[:, actions]
                    y_selected = y[actions]

                    if x_selected.shape[1] == 0:
                        continue

                    n_samples += x_selected.shape[1]
                    total_n_samples += x_selected.shape[1]

                    # Update the IMDB network with selected data
                    cost = model.f_train(x_selected, mask_selected, y_selected)

                    if iteration % PolicyConfig['AC_update_freq'] == 0:
                        # Get immediate reward
                        if PolicyConfig['cost_gap_AC_reward']:
                            cost_old = cost
                            cost_new = model.f_cost(x, mask, y)
                            imm_reward = cost_old - cost_new
                        else:
                            valid_part_x, valid_part_y = get_part_data(
                                np.asarray(valid_x), np.asarray(valid_y), PolicyConfig['immediate_reward_sample_size'])
                            valid_err = model.predict_error(valid_part_x, valid_part_y, kf_valid_part)
                            imm_reward = 1. - valid_err

                        # Get new state, new actions, and compute new Q value
                        probability_new = model.get_policy_input(x, mask, y, epoch, history_accuracy)
                        actions_new = actor.take_action(probability_new, log_replay=False)

                        Q_value_new = critic.Q_function(state=probability_new, action=actions_new)
                        if epoch < ParamConfig['epoch_per_episode'] - 1:
                            label = PolicyConfig['actor_gamma'] * Q_value_new + imm_reward
                        else:
                            label = imm_reward

                        # Update the critic Q network
                        Q_loss = critic.update(probability, actions, floatX(label))

                        # Update actor network
                        actor_loss = actor.update_raw(probability, actions,
                                                      np.full(actions.shape, label, dtype=probability.dtype))

                        if PolicyConfig['AC_update_freq'] >= display_freq or iteration % display_freq == 0:
                            message('\tCritic Q network loss', Q_loss, end='')
                            message('\tActor network loss', actor_loss)

                    if cost is not None and (np.isnan(cost) or np.isinf(cost)):
                        message('bad cost detected: ', cost)
                        return 1., 1., 1.

                    if iteration % display_freq == 0:
                        message('Epoch', epoch, '\tUpdate', iteration, '\tCost', cost, end='')

                    # Do not save when training policy!

                    if iteration % ParamConfig['train_loss_freq'] == 0:
                        train_loss = model.get_training_loss(train_x, train_y)
                        message('Training Loss:', train_loss)

                    if iteration % valid_freq == 0:
                        model.use_noise.set_value(0.)
                        # train_err = model.predict_error(train_x, train_y, kf)
                        valid_err = model.predict_error(valid_x, valid_y, kf_valid)
                        test_err = model.predict_error(test_x, test_y, kf_test)

                        history_errs.append([valid_err, test_err])
                        history_accuracy.append(1. - valid_err)

                        if best_parameters is None or valid_err <= np.array(history_errs)[:, 0].min():
                            best_parameters = model.get_parameter_values()
                            bad_counter = 0

                        message('Train', 0.00, 'Valid', valid_err, 'Test', test_err,
                                'Total_samples', total_n_samples)

                        if len(history_errs) > patience and valid_err >= np.array(history_errs)[:-patience, 0].min():
                            bad_counter += 1
                            if bad_counter > patience:
                                print('Early Stop!')
                                early_stop = True
                                break

                message('Seen %d samples' % n_samples)

                if early_stop:
                    break
        except KeyboardInterrupt:
            message('Training interrupted')

        end_time = time.time()

        if Config['policy_save_freq'] > 0 and episode % Config['policy_save_freq'] == 0:
            actor.save_policy()

        train_err, valid_err, test_err = test_and_post_process(
            model,
            train_small_size, train_small_x, train_small_y, valid_x, valid_y, test_x,
            test_y,
            kf_valid, kf_test,
            history_errs, best_parameters,
            epoch, start_time, end_time,
            save_to=False)


def test_policy_IMDB():
    np.random.seed(ParamConfig['seed'])

    # Loading data
    train_x, train_y, valid_x, valid_y, test_x, test_y, \
        train_size, valid_size, test_size = pre_process_data()

    # Building model
    model = IMDBModel(ParamConfig['reload_model'])

    # Loading configure settings
    kf_valid, kf_test, \
        valid_freq, save_freq, display_freq, \
        save_to, patience = pre_process_config(model, train_size, valid_size, test_size)

    if Config['train_type'] == 'random_drop':
        # Random drop configure
        random_drop_numbers = map(lambda l: int(l.strip()), list(open(ParamConfig['random_drop_number_file'], 'r')))
        random_drop_index = 0
    else:
        # Build policy
        input_size = model.get_policy_input_size()
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
    epoch = 0
    history_accuracy = []
    start_time = time.time()

    try:
        total_n_samples = 0

        for epoch in range(ParamConfig['epoch_per_episode']):
            print('[Epoch {}]'.format(epoch))
            message('[Epoch {}]'.format(epoch))

            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(train_size, model.train_batch_size, shuffle=True)

            for _, train_index in kf:
                update_index += 1
                model.use_noise.set_value(floatX(1.))

                # Select the random examples for this minibatch
                x = [train_x[t] for t in train_index]
                y = [train_y[t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, np.asarray(y, dtype='int64'))

                # Policy take action here
                probability = model.get_policy_input(x, mask, y, epoch, history_accuracy)

                if Config['train_type'] == 'deterministic':
                    raise NotImplementedError('Deterministic is not implemented')
                else:
                    if Config['train_type'] == 'stochastic':
                        actions = policy.take_action(probability, False)
                    elif Config['train_type'] == 'random_drop':
                        if random_drop_index >= len(random_drop_numbers):
                            rate = 1.
                        else:
                            rate = 1. - float(random_drop_numbers[random_drop_index]) / (
                                valid_freq * model.train_batch_size),

                        actions = np.random.binomial(1, rate, y.shape).astype(bool)

                    # get masked inputs and targets
                    x = x[:, actions]
                    mask = mask[:, actions]
                    y = y[actions]

                    n_samples += x.shape[1]
                    total_n_samples += x.shape[1]

                    cost = model.f_train(x, mask, y)

                if cost is not None and np.isnan(cost) or np.isinf(cost):
                    message('bad cost detected: ', cost)
                    return 1., 1., 1.

                if update_index % display_freq == 0:
                    message('Epoch ', epoch, 'Update ', update_index, 'Cost ', cost)

                if save_to and update_index % save_freq == 0:
                    save_parameters(model, best_parameters, save_to, history_errs)

                if update_index % ParamConfig['train_loss_freq'] == 0:
                    train_loss = model.get_training_loss(train_x, train_y)
                    message('Training Loss:', train_loss)

                if update_index % valid_freq == 0:
                    model.use_noise.set_value(0.)
                    # train_err = model.predict_error(train_x, train_y, kf)
                    valid_err = model.predict_error(valid_x, valid_y, kf_valid)
                    test_err = model.predict_error(test_x, test_y, kf_test)

                    history_errs.append([valid_err, test_err])
                    history_accuracy.append(1. - valid_err)

                    if best_parameters is None or valid_err <= np.array(history_errs)[:, 0].min():
                        best_parameters = model.get_parameter_values()
                        bad_counter = 0

                    message('Train', 0.0, 'Valid', valid_err, 'Test', test_err,
                            'Total_samples', total_n_samples)

                    if Config['train_type'] == 'random_drop':
                        random_drop_index += 1

                    if len(history_errs) > patience and valid_err >= np.array(history_errs)[:-patience, 0].min():
                        bad_counter += 1
                        if bad_counter > patience:
                            message('Early Stop!')
                            early_stop = True
                            break

            message('Seen %d samples' % n_samples)

            if early_stop:
                break
    except KeyboardInterrupt:
        message('Training interrupted')

    end_time = time.time()

    test_and_post_process(model,
                          train_size, train_x, train_y, valid_x, valid_y, test_x, test_y,
                          kf_valid, kf_test,
                          history_errs, best_parameters,
                          epoch, start_time, end_time,
                          save_to=False)


if __name__ == '__main__':
    process_before_train(ParamConfig)

    try:
        if Config['train_type'] == 'raw':
            train_raw_IMDB()
        elif Config['train_type'] == 'self_paced':
            train_SPL_IMDB()
        elif Config['train_type'] == 'policy':
            train_policy_IMDB()
        elif Config['train_type'] == 'actor_critic':
            train_actor_critic_IMDB()
        elif Config['train_type'] == 'deterministic':
            test_policy_IMDB()
        elif Config['train_type'] == 'stochastic':
            test_policy_IMDB()
        elif Config['train_type'] == 'random_drop':
            test_policy_IMDB()
        else:
            raise Exception('Unknown train type {}'.format(Config['train_type']))
    except:
        message(traceback.format_exc())
    finally:
        finalize_logging_file()
