# -*- coding: utf-8 -*-

from __future__ import print_function

from ..batch_updater import *
from ..critic_network import CriticNetwork
from ..model_class.IMDB import IMDBModel
from ..policy_network import PolicyNetworkBase
from ..reward_checker import get_reward_checker, RewardChecker
from ..utility.IMDB import pre_process_IMDB_data, pre_process_config
from ..utility.IMDB import prepare_imdb_data as prepare_data
from ..utility.utils import *
from ..utility.config import IMDBConfig as ParamConfig, Config


def save_parameters(model, best_parameters, save_to, history_errs):
    message('Saving model parameters...', end=' ')

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

        if Config['temp_job'] in RemainOrderJobs:
            x_train_small, y_train_small = x_train, y_train
        else:
            # get small training data
            x_train_small, y_train_small = get_part_data(x_train, y_train, ParamConfig['train_small_size'])
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

            kf = get_minibatches_idx(train_small_size, model.train_batch_size, shuffle=True)

            for _, train_index in kf:
                model.use_noise.set_value(floatX(1.))
                part_train_cost = updater.add_batch(train_index)

                if updater.total_train_batches > 0 and \
                        updater.total_train_batches != last_validate_point and \
                        updater.total_train_batches % valid_freq == 0:
                    last_validate_point = updater.total_train_batches

                    model.use_noise.set_value(floatX(0.))
                    validate_acc, test_acc = validate_point_message(
                        model, x_train, y_train, x_validate, y_validate, x_test, y_test, updater, reward_checker,
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


def train_actor_critic_IMDB():
    # Loading data
    x_train, y_train, x_valid, y_valid, x_test, y_test, \
        train_size, valid_size, test_size = pre_process_IMDB_data()

    # Building model
    model = IMDBModel(ParamConfig['reload_model'])

    # Loading configure settings
    kf_valid, kf_test, \
        valid_freq, save_freq, display_freq, \
        save_to, patience = pre_process_config(model, train_size, valid_size, test_size)

    # Build Actor and Critic network
    input_size = model.get_policy_input_size()
    message('Input size of policy network:', input_size)
    actor = PolicyNetworkBase.get_by_name(PolicyConfig['policy_model_type'])(input_size=input_size)
    critic = CriticNetwork(feature_size=input_size, batch_size=model.train_batch_size)

    actor.check_load()

    # Train the network
    start_episode = 1 + PolicyConfig['start_episode']
    for episode in range(start_episode, start_episode + PolicyConfig['num_episodes']):
        start_new_episode(model, actor, episode)

        # Train the network
        # Some variables

        # To prevent the double validate / AC update point
        last_validate_point = -1
        last_AC_update_point = -1

        if Config['temp_job'] in RemainOrderJobs:
            x_train_small, y_train_small = x_train, y_train
        else:
            # get small training data
            x_train_small, y_train_small = get_part_data(x_train, y_train, ParamConfig['train_small_size'])
        train_small_size = len(x_train_small)
        message('Training small size:', train_small_size)

        updater = ACUpdater(model, [x_train_small, y_train_small], actor, prepare_data=prepare_data)

        best_validate_acc = -np.inf
        best_iteration = 0
        test_score = 0.0
        bad_counter = 0
        early_stop = False
        start_time = time.time()

        for epoch in range(ParamConfig['epoch_per_episode']):
            epoch_start_time = start_new_epoch(updater, epoch)

            kf = get_minibatches_idx(train_small_size, model.train_batch_size, shuffle=True)

            for _, train_index in kf:
                part_train_cost = updater.add_batch(train_index)

                if updater.total_train_batches > 0 and \
                        updater.total_train_batches != last_AC_update_point and \
                        updater.total_train_batches % PolicyConfig['AC_update_freq'] == 0:
                    last_AC_update_point = updater.total_train_batches

                    # [NOTE]: The batch is the batch sent into updater, NOT the buffer's batch.
                    inputs = x_train_small[train_index]
                    targets = y_train_small[train_index]
                    x, mask, y = prepare_data(inputs, targets)
                    probability = updater.last_probability
                    actions = updater.last_action

                    # Get immediate reward
                    valid_part_x, valid_part_y = get_part_data(
                        np.asarray(x_valid), np.asarray(y_valid), PolicyConfig['immediate_reward_sample_size'])
                    _, valid_acc, validate_batches = model.validate_or_test(valid_part_x, valid_part_y)
                    imm_reward = valid_acc / validate_batches

                    # Get new state, new actions, and compute new Q value
                    probability_new = model.get_policy_input(x, mask, y, updater, updater.history_accuracy)
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

                    if PolicyConfig['AC_update_freq'] >= display_freq or \
                            updater.total_train_batches % display_freq == 0:
                        message('E {} TB {} Cost {:.6f} Critic loss {:.6f} Actor loss {:.6f}'
                                .format(epoch, updater.total_train_batches, part_train_cost, Q_loss, actor_loss))

                if updater.total_train_batches > 0 and \
                        updater.total_train_batches != last_validate_point and \
                        updater.total_train_batches % valid_freq == 0:
                    last_validate_point = updater.total_train_batches

                    validate_acc, test_acc = validate_point_message(
                        model, x_train, y_train, x_valid, y_valid, x_test, y_test, updater)

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

        if PolicyConfig['policy_save_freq'] > 0 and episode % PolicyConfig['policy_save_freq'] == 0:
            actor.save_policy(PolicyConfig['policy_save_file'], episode)


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
    })
