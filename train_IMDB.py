# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import heapq
from collections import deque

from config import IMDBConfig as ParamConfig
from critic_network import CriticNetwork
from model_IMDB import IMDBModel
from policy_network import PolicyNetworkBase
from utils import *
from utils_IMDB import pre_process_IMDB_data, pre_process_config
from utils_IMDB import prepare_imdb_data as prepare_data
from batch_updater import *
from reward_checker import get_reward_checker, RewardChecker

__author__ = 'fyabc'


# TODO: Change code into updaters
# raw: Done
# spl: Done
# REINFORCE: Done
# A-C: X
# Test: Done


def save_parameters(model, best_parameters, save_to, history_errs):
    message('Saving model parameters...')

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
    # Loading data
    # [NOTE] This must before the build of model, because the ParamConfig['ydim'] should be set.
    x_train, y_train, x_validate, y_validate, x_test, y_test, \
        train_size, valid_size, test_size = pre_process_IMDB_data()

    model = IMDBModel()

    # Loading configure settings
    kf_valid, kf_test, \
        valid_freq, save_freq, display_freq, \
        save_to, patience = pre_process_config(model, train_size, valid_size, test_size)

    updater = RawUpdater(model, [x_train, y_train], prepare_data=prepare_data)

    # Train the network
    # Some variables
    # To prevent the double validate point
    last_validate_point = -1

    best_validate_acc = -np.inf
    best_iteration = 0
    test_score = 0.0
    bad_counter = 0
    early_stop = False
    start_time = time.time()

    for epoch in range(ParamConfig['epoch_per_episode']):
        epoch_start_time = start_new_epoch(updater, epoch)

        kf = get_minibatches_idx(train_size, model.train_batch_size, shuffle=True)

        for _, train_index in kf:
            model.use_noise.set_value(floatX(1.))
            part_train_cost = updater.add_batch(train_index)

            # Log training loss of each batch in test process
            if part_train_cost is not None:
                message("tL {}: {:.6f}".format(updater.epoch_train_batches, part_train_cost.tolist()))

            if updater.total_train_batches > 0 and \
                    updater.total_train_batches != last_validate_point and \
                    updater.total_train_batches % valid_freq == 0:
                last_validate_point = updater.total_train_batches

                model.use_noise.set_value(floatX(0.))
                validate_acc, test_acc = validate_point_message(
                    model, x_train, y_train, x_validate, y_validate, x_test, y_test, updater,
                    validate_size=valid_size,  # Use part validation set in baseline
                    run_test=True,
                )

                if validate_acc > best_validate_acc:
                    best_validate_acc = validate_acc
                    best_iteration = updater.total_train_batches
                    test_score = test_acc
                    bad_counter = 0

                if len(updater.history_accuracy) > patience and \
                        validate_acc <= max(updater.history_accuracy[:-patience]):
                    bad_counter += 1
                    if bad_counter > patience:
                        early_stop = True
                        break

        message("Epoch {} of {} took {:.3f}s".format(
            epoch, ParamConfig['epoch_per_episode'], time.time() - epoch_start_time))
        if early_stop:
            message('Early Stop!')
            break

    episode_final_message(best_validate_acc, best_iteration, test_score, start_time)


def train_SPL_IMDB():
    # Loading data
    # [NOTE] This must before the build of model, because the ParamConfig['ydim'] should be set.
    x_train, y_train, x_validate, y_validate, x_test, y_test, \
        train_size, valid_size, test_size = pre_process_IMDB_data()

    model = IMDBModel()

    # Loading configure settings
    kf_valid, kf_test, \
        valid_freq, save_freq, display_freq, \
        save_to, patience = pre_process_config(model, train_size, valid_size, test_size)

    updater = SPLUpdater(model, [x_train, y_train], ParamConfig['epoch_per_episode'], prepare_data=prepare_data)

    # Train the network
    # Some variables
    # To prevent the double validate point
    last_validate_point = -1

    best_validate_acc = -np.inf
    best_iteration = 0
    test_score = 0.0
    bad_counter = 0
    early_stop = False
    start_time = time.time()

    for epoch in range(ParamConfig['epoch_per_episode']):
        epoch_start_time = start_new_epoch(updater, epoch)

        kf = get_minibatches_idx(train_size, model.train_batch_size, shuffle=True)

        for _, train_index in kf:
            model.use_noise.set_value(floatX(1.))
            part_train_cost = updater.add_batch(train_index)

            # Log training loss of each batch in test process
            if part_train_cost is not None:
                message("tL {}: {:.6f}".format(updater.epoch_train_batches, part_train_cost.tolist()))

            if updater.total_train_batches > 0 and \
                    updater.total_train_batches != last_validate_point and \
                    updater.total_train_batches % valid_freq == 0:
                last_validate_point = updater.total_train_batches

                model.use_noise.set_value(floatX(0.))
                validate_acc, test_acc = validate_point_message(
                    model, x_train, y_train, x_validate, y_validate, x_test, y_test, updater,
                    validate_size=valid_size,  # Use part validation set in baseline
                    run_test=True,
                )

                if validate_acc > best_validate_acc:
                    best_validate_acc = validate_acc
                    best_iteration = updater.total_train_batches
                    test_score = test_acc
                    bad_counter = 0

                if len(updater.history_accuracy) > patience and \
                        validate_acc <= max(updater.history_accuracy[:-patience]):
                    bad_counter += 1
                    if bad_counter > patience:
                        early_stop = True
                        break

        message("Epoch {} of {} took {:.3f}s".format(
            epoch, ParamConfig['epoch_per_episode'], time.time() - epoch_start_time))
        if early_stop:
            message('Early Stop!')
            break

    episode_final_message(best_validate_acc, best_iteration, test_score, start_time)


def train_policy_IMDB():
    # Loading data
    # [NOTE] This must before the build of model, because the ParamConfig['ydim'] should be set.
    x_train, y_train, x_validate, y_validate, x_test, y_test, \
        train_size, valid_size, test_size = pre_process_IMDB_data()

    model = IMDBModel()

    # Build policy
    input_size = model.get_policy_input_size()
    message('Input size of policy network:', input_size)
    policy = PolicyNetworkBase.get_by_name(PolicyConfig['policy_model_type'])(input_size=input_size)
    # policy = LRPolicyNetwork(input_size=input_size)

    policy.check_load()

    # Loading configure settings
    kf_valid, kf_test, \
        valid_freq, save_freq, display_freq, \
        save_to, patience = pre_process_config(model, train_size, valid_size, test_size)

    reward_checker_type = RewardChecker.get_by_name(PolicyConfig['reward_checker'])

    start_episode = 1 + PolicyConfig['start_episode']
    for episode in range(start_episode, start_episode + PolicyConfig['num_episodes']):
        start_new_episode(model, policy, episode)

        # Train the network
        # Some variables

        # To prevent the double validate point
        last_validate_point = -1

        # get small training data
        # x_train_small, y_train_small = get_part_data(x_train, y_train, ParamConfig['train_small_size'])
        # [WARNING] Do NOT shuffle here!!!
        x_train_small, y_train_small = x_train, y_train

        train_small_size = len(x_train_small)
        message('Training small size:', train_small_size)

        # Speed reward
        reward_checker = get_reward_checker(
            reward_checker_type,
            ParamConfig['epoch_per_episode'] * train_small_size
        )

        updater = TrainPolicyUpdater(model, [x_train_small, y_train_small], policy, prepare_data=prepare_data)

        best_validate_acc = -np.inf
        best_iteration = 0
        test_score = 0.0
        bad_counter = 0
        early_stop = False
        start_time = time.time()

        for epoch in range(ParamConfig['epoch_per_episode']):
            epoch_start_time = start_new_epoch(updater, epoch)

            kf = get_minibatches_idx(train_size, model.train_batch_size, shuffle=True)

            for _, train_index in kf:
                model.use_noise.set_value(floatX(1.))
                part_train_cost = updater.add_batch(train_index)

                if updater.total_train_batches > 0 and \
                        updater.total_train_batches != last_validate_point and \
                        updater.total_train_batches % valid_freq == 0:
                    last_validate_point = updater.total_train_batches

                    model.use_noise.set_value(floatX(0.))
                    validate_acc, test_acc = validate_point_message(
                        model, x_train, y_train, x_validate, y_validate, x_test, y_test, updater,
                        validate_size=valid_size,  # Use part validation set in baseline
                        run_test=False,
                    )

                    if validate_acc > best_validate_acc:
                        best_validate_acc = validate_acc
                        best_iteration = updater.total_train_batches
                        test_score = test_acc
                        bad_counter = 0

                    if len(updater.history_accuracy) > patience and \
                            validate_acc <= max(updater.history_accuracy[:-patience]):
                        bad_counter += 1
                        if bad_counter > patience:
                            early_stop = True
                            break

            message("Epoch {} of {} took {:.3f}s".format(
                epoch, ParamConfig['epoch_per_episode'], time.time() - epoch_start_time))
            if early_stop:
                message('Early Stop!')
                break

        episode_final_message(best_validate_acc, best_iteration, test_score, start_time)

        policy.update(reward_checker)

        if PolicyConfig['policy_save_freq'] > 0 and episode % PolicyConfig['policy_save_freq'] == 0:
            policy.save_policy(PolicyConfig['policy_save_file'], episode)


def train_actor_critic_IMDB_old():
    # Loading data
    train_x, train_y, valid_x, valid_y, test_x, test_y, \
        train_size, valid_size, test_size = pre_process_IMDB_data()

    # Building model
    model = IMDBModel(ParamConfig['reload_model'])

    # Loading configure settings
    kf_valid, kf_test, \
        valid_freq, save_freq, display_freq, \
        save_to, patience = pre_process_config(model, train_size, valid_size, test_size)

    kf_valid_part = get_minibatches_idx(PolicyConfig['immediate_reward_sample_size'], model.validate_batch_size)

    # Build Actor and Critic network
    input_size = model.get_policy_input_size()
    message('Input size of policy network:', input_size)
    actor = PolicyNetworkBase.get_by_name(PolicyConfig['policy_model_type'])(input_size=input_size)
    # actor = LRPolicyNetwork(input_size=input_size)
    critic = CriticNetwork(feature_size=input_size, batch_size=model.train_batch_size)

    actor.check_load()

    start_episode = 1 + PolicyConfig['start_episode']
    for episode in range(start_episode, start_episode + PolicyConfig['num_episodes']):
        print('[Episode {}]'.format(episode))
        message('[Episode {}]'.format(episode))

        actor.message_parameters()

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

        end_time = time.time()

        if PolicyConfig['policy_save_freq'] > 0 and episode % PolicyConfig['policy_save_freq'] == 0:
            actor.save_policy(PolicyConfig['policy_save_file'], episode)

        train_err, valid_err, test_err = test_and_post_process(
            model,
            train_small_size, train_small_x, train_small_y, valid_x, valid_y, test_x,
            test_y,
            kf_valid, kf_test,
            history_errs, best_parameters,
            epoch, start_time, end_time,
            save_to=False)


def train_actor_critic_IMDB():
    pass


def test_policy_IMDB():
    # Loading data
    # [NOTE] This must before the build of model, because the ParamConfig['ydim'] should be set.
    x_train, y_train, x_validate, y_validate, x_test, y_test, \
        train_size, valid_size, test_size = pre_process_IMDB_data()

    model = IMDBModel()

    # Loading configure settings
    kf_valid, kf_test, \
        valid_freq, save_freq, display_freq, \
        save_to, patience = pre_process_config(model, train_size, valid_size, test_size)

    if Config['train_type'] == 'random_drop':
        updater = RandomDropUpdater(model, [x_train, y_train],
                                    PolicyConfig['random_drop_number_file'], prepare_data=prepare_data,
                                    drop_num_type='vp', valid_freq=ParamConfig['valid_freq'])
    else:
        input_size = IMDBModel.get_policy_input_size()
        message('Input size of policy network:', input_size)

        # Build policy
        policy = PolicyNetworkBase.get_by_name(PolicyConfig['policy_model_type'])(input_size=input_size)
        # policy = LRPolicyNetwork(input_size=input_size)
        policy.load_policy()
        policy.message_parameters()
        updater = TestPolicyUpdater(model, [x_train, y_train], policy, prepare_data=prepare_data)

    # Train the network
    # Some variables
    # To prevent the double validate point
    last_validate_point = -1

    best_validate_acc = -np.inf
    best_iteration = 0
    test_score = 0.0
    bad_counter = 0
    early_stop = False
    start_time = time.time()

    for epoch in range(ParamConfig['epoch_per_episode']):
        epoch_start_time = start_new_epoch(updater, epoch)

        kf = get_minibatches_idx(train_size, model.train_batch_size, shuffle=True)

        for _, train_index in kf:
            model.use_noise.set_value(floatX(1.))
            part_train_cost = updater.add_batch(train_index)

            # Log training loss of each batch in test process
            if part_train_cost is not None:
                message("tL {}: {:.6f}".format(updater.epoch_train_batches, part_train_cost.tolist()))

            if updater.total_train_batches > 0 and \
                    updater.total_train_batches != last_validate_point and \
                    updater.total_train_batches % valid_freq == 0:
                last_validate_point = updater.total_train_batches

                model.use_noise.set_value(floatX(0.))
                validate_acc, test_acc = validate_point_message(
                    model, x_train, y_train, x_validate, y_validate, x_test, y_test, updater,
                    # validate_size=valid_size,  # Use part validation set in baseline
                    run_test=True,
                )

                if validate_acc > best_validate_acc:
                    best_validate_acc = validate_acc
                    best_iteration = updater.total_train_batches
                    test_score = test_acc
                    bad_counter = 0

                if len(updater.history_accuracy) > patience and \
                        validate_acc <= max(updater.history_accuracy[:-patience]):
                    bad_counter += 1
                    if bad_counter > patience:
                        early_stop = True
                        break

        message("Epoch {} of {} took {:.3f}s".format(
            epoch, ParamConfig['epoch_per_episode'], time.time() - epoch_start_time))
        if early_stop:
            message('Early Stop!')
            break

    episode_final_message(best_validate_acc, best_iteration, test_score, start_time)


def new_train_IMDB():
    pass
        
        
def main():
    dataset_main({
        'raw': train_raw_IMDB,
        'self_paced': train_SPL_IMDB,
        'spl': train_SPL_IMDB,

        'policy': train_policy_IMDB,
        'reinforce': train_policy_IMDB,
        'speed': train_policy_IMDB,

        'actor_critic': train_actor_critic_IMDB,
        'ac': train_actor_critic_IMDB,

        # 'test': test_policy_IMDB,
        'deterministic': test_policy_IMDB,
        'stochastic': test_policy_IMDB,
        'random_drop': test_policy_IMDB,

        'new_train': new_train_IMDB,
    })


if __name__ == '__main__':
    main()
