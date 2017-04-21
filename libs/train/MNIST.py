#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from functools import partial

from ..batch_updater import *
from ..critic_network import CriticNetwork
from ..model_class.MNIST import MNISTModel
from ..new_train.new_train_MNIST import new_train_MNIST
from ..policy_network import PolicyNetworkBase
from ..reward_checker import RewardChecker, get_reward_checker
from ..utility.MNIST import pre_process_MNIST_data, pre_process_config
from ..utility.config import MNISTConfig as ParamConfig
from ..utility.utils import *

__author__ = 'fyabc'


def train_raw_MNIST_template(train_type='raw'):
    model = MNISTModel()

    # Load the dataset and config
    x_train, y_train, x_validate, y_validate, x_test, y_test,\
        train_size, validate_size, test_size = pre_process_MNIST_data()
    patience, patience_increase, improvement_threshold, validation_frequency = pre_process_config(model, train_size)

    #####################################
    # Train type specific configuration #
    #####################################

    epoch_shuffle = True

    if train_type == 'raw':
        updater = RawUpdater(model, [x_train, y_train])
        epoch_shuffle = ParamConfig['raw_shuffle']
    elif train_type == 'spl':
        updater = SPLUpdater(model, [x_train, y_train], ParamConfig['epoch_per_episode'])
    elif train_type == 'deterministic':
        raise NotImplementedError('Deterministic test policy is not implemented in MNIST')
    elif train_type == 'random_drop':
        updater = RandomDropUpdater(model, [x_train, y_train], PolicyConfig['random_drop_number_file'],
                                    drop_num_type='vp', valid_freq=ParamConfig['valid_freq'])
    elif train_type == 'stochastic':
        input_size = MNISTModel.get_policy_input_size()
        message('Input size of policy network:', input_size)

        # Build policy
        policy = PolicyNetworkBase.get_by_name(PolicyConfig['policy_model_type'])(input_size=input_size)
        # policy = LRPolicyNetwork(input_size=input_size)
        policy.load_policy()
        policy.message_parameters()
        updater = TestPolicyUpdater(model, [x_train, y_train], policy)
    else:
        raise KeyError('Unknown train type {}'.format(train_type))

    #####################
    # Configuration end #
    #####################

    # Train the network
    # Some variables

    # To prevent the double validate point
    last_validate_point = -1

    best_validate_acc = -np.inf
    best_iteration = 0
    test_score = 0.0
    start_time = time.time()

    for epoch in range(ParamConfig['epoch_per_episode']):
        epoch_start_time = start_new_epoch(updater, epoch)

        kf = get_minibatches_idx(train_size, model.train_batch_size, shuffle=epoch_shuffle)

        for _, train_index in kf:
            part_train_cost = updater.add_batch(train_index)

            # Log training loss of each batch in test process
            if part_train_cost is not None:
                message("tL {}: {:.6f}".format(updater.epoch_train_batches, part_train_cost.tolist()))

            if updater.total_train_batches > 0 and \
                    updater.total_train_batches != last_validate_point and \
                    updater.total_train_batches % validation_frequency == 0:
                last_validate_point = updater.total_train_batches
                validate_acc, test_acc = validate_point_message(
                    model, x_train, y_train, x_validate, y_validate, x_test, y_test, updater,
                    # validate_size=validate_size,  # Use part validation set in baseline
                    run_test=True,
                )

                if validate_acc > best_validate_acc:
                    # improve patience if loss improvement is good enough
                    if (1. - validate_acc) < (1. - best_validate_acc) * improvement_threshold:
                        patience = max(patience, updater.total_train_batches * patience_increase)
                    best_validate_acc = validate_acc
                    best_iteration = updater.total_train_batches
                    test_score = test_acc

            if updater.total_train_batches >= patience:
                break

        message("Epoch {} of {} took {:.3f}s".format(
            epoch, ParamConfig['epoch_per_episode'], time.time() - epoch_start_time))
        if updater.total_train_batches >= patience:
            message('Early Stop!')
            break

    episode_final_message(best_validate_acc, best_iteration, test_score, start_time)

    if train_type == 'raw':
        if ParamConfig['save_model']:
            model.save_model()

train_raw_MNIST = partial(train_raw_MNIST_template, 'raw')
train_SPL_MNIST = partial(train_raw_MNIST_template, 'spl')
test_stochastic_MNIST = partial(train_raw_MNIST_template, 'stochastic')
test_deterministic_MNIST = partial(train_raw_MNIST_template, 'deterministic')
test_random_drop_MNIST = partial(train_raw_MNIST_template, 'random_drop')


def train_policy_MNIST():
    model = MNISTModel()

    if ParamConfig['warm_start']:
        model.load_model()

    # Create the policy network
    input_size = MNISTModel.get_policy_input_size()
    message('Input size of policy network:', input_size)
    policy = PolicyNetworkBase.get_by_name(PolicyConfig['policy_model_type'])(input_size=input_size)
    # policy = LRPolicyNetwork(input_size=input_size)

    policy.check_load()

    # Load the dataset and config
    x_train, y_train, x_validate, y_validate, x_test, y_test,\
        train_size, validate_size, test_size = pre_process_MNIST_data()
    patience, patience_increase, improvement_threshold, validation_frequency = pre_process_config(model, train_size)

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

        updater = TrainPolicyUpdater(model, [x_train_small, y_train_small], policy)

        best_validate_acc = -np.inf
        best_iteration = 0
        test_score = 0.0
        start_time = time.time()

        for epoch in range(ParamConfig['epoch_per_episode']):
            epoch_start_time = start_new_epoch(updater, epoch)

            kf = get_minibatches_idx(train_small_size, model.train_batch_size, shuffle=True)

            for _, train_index in kf:
                part_train_cost = updater.add_batch(train_index)

                if updater.total_train_batches > 0 and \
                        updater.total_train_batches != last_validate_point and \
                        updater.total_train_batches % validation_frequency == 0:
                    last_validate_point = updater.total_train_batches
                    validate_acc, test_acc = validate_point_message(
                        model, x_train, y_train, x_validate, y_validate, x_test, y_test, updater, reward_checker,
                        run_test=False,
                    )
                    
                    if validate_acc > best_validate_acc:
                        # improve patience if loss improvement is good enough
                        if (1. - validate_acc) < (1. - best_validate_acc) * improvement_threshold:
                            patience = max(patience, updater.total_train_batches * patience_increase)
                        best_validate_acc = validate_acc
                        best_iteration = updater.total_train_batches
                        test_score = test_acc

                if updater.total_train_batches >= patience:
                    break

            message("Epoch {} of {} took {:.3f}s".format(
                epoch, ParamConfig['epoch_per_episode'], time.time() - epoch_start_time))

            if updater.total_train_batches >= patience:
                message('Early Stop!')
                break

        episode_final_message(best_validate_acc, best_iteration, test_score, start_time)

        policy.update(reward_checker)

        if PolicyConfig['policy_save_freq'] > 0 and episode % PolicyConfig['policy_save_freq'] == 0:
            policy.save_policy(PolicyConfig['policy_save_file'], episode)


def train_actor_critic_MNIST():
    model = MNISTModel()

    # Create the policy network
    input_size = MNISTModel.get_policy_input_size()
    message('Input size of policy network:', input_size)
    actor = PolicyNetworkBase.get_by_name(PolicyConfig['policy_model_type'])(input_size=input_size)

    actor.check_load()

    # actor = LRPolicyNetwork(input_size=input_size)
    critic = CriticNetwork(feature_size=input_size, batch_size=model.train_batch_size)

    # Load the dataset and config
    x_train, y_train, x_validate, y_validate, x_test, y_test, train_size, validate_size, test_size = pre_process_MNIST_data()
    patience, patience_increase, improvement_threshold, validation_frequency = pre_process_config(model, train_size)

    # Train the network
    start_episode = 1 + PolicyConfig['start_episode']
    for episode in range(start_episode, start_episode + PolicyConfig['num_episodes']):
        start_new_episode(model, actor, episode)

        # Train the network
        # Some variables

        # To prevent the double validate / AC update point
        last_validate_point = -1
        last_AC_update_point = -1

        # get small training data
        # x_train_small, y_train_small = get_part_data(x_train, y_train, ParamConfig['train_small_size'])
        # [WARNING] Do NOT shuffle here!!!
        x_train_small, y_train_small = x_train, y_train
        train_small_size = len(x_train_small)
        message('Training small size:', train_small_size)

        updater = ACUpdater(model, [x_train_small, y_train_small], actor)

        best_validate_acc = -np.inf
        best_iteration = 0
        test_score = 0.0
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
                    probability = updater.last_probability
                    actions = updater.last_action

                    # Get immediate reward
                    # [NOTE]: Cost gap reward is removed
                    # if PolicyConfig['cost_gap_AC_reward']:
                    #     cost_old = part_train_cost
                    #
                    #     cost_new = model.f_cost_without_decay(inputs, targets)
                    #     imm_reward = cost_old - cost_new
                    valid_part_x, valid_part_y = get_part_data(
                        np.asarray(x_validate), np.asarray(y_validate), PolicyConfig['immediate_reward_sample_size'])
                    _, valid_acc, validate_batches = model.validate_or_test(valid_part_x, valid_part_y)
                    imm_reward = valid_acc / validate_batches

                    # Get new state, new actions, and compute new Q value
                    probability_new = model.get_policy_input(inputs, targets, updater, updater.history_accuracy)
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

                    if PolicyConfig['AC_update_freq'] >= ParamConfig['display_freq'] or \
                       updater.total_train_batches % ParamConfig['display_freq'] == 0:
                        message('Epoch {}\tTotalBatches {}\tCost {}\tCritic loss {}\tActor loss {}'
                                .format(epoch, updater.total_train_batches, part_train_cost, Q_loss, actor_loss))

                if updater.total_train_batches > 0 and \
                        updater.total_train_batches != last_validate_point and \
                        updater.total_train_batches % validation_frequency == 0:
                    last_validate_point = updater.total_train_batches

                    validate_acc, test_acc = validate_point_message(
                        model, x_train, y_train, x_validate, y_validate, x_test, y_test, updater)
                    
                    if validate_acc > best_validate_acc:
                        # improve patience if loss improvement is good enough
                        if (1. - validate_acc) < (1. - best_validate_acc) * improvement_threshold:
                            patience = max(patience, updater.total_train_batches * patience_increase)
                        best_validate_acc = validate_acc
                        best_iteration = updater.total_train_batches
                        test_score = test_acc

                if updater.total_train_batches >= patience:
                    break

            message("Epoch {} of {} took {:.3f}s".format(
                epoch, ParamConfig['epoch_per_episode'], time.time() - epoch_start_time))
            if updater.total_train_batches >= patience:
                message('Early Stop!')
                break

        episode_final_message(best_validate_acc, best_iteration, test_score, start_time)

        # [NOTE]: Remove update of terminal reward in AC.
        # validate_acc = model.get_test_acc(x_validate, y_validate)
        # actor.update(validate_acc)

        if PolicyConfig['policy_save_freq'] > 0 and episode % PolicyConfig['policy_save_freq'] == 0:
            actor.save_policy(PolicyConfig['policy_save_file'], episode)


def main():
    dataset_main({
        'raw': train_raw_MNIST,
        'self_paced': train_SPL_MNIST,
        'spl': train_SPL_MNIST,

        'policy': train_policy_MNIST,
        'reinforce': train_policy_MNIST,
        'speed': train_policy_MNIST,

        'actor_critic': train_actor_critic_MNIST,
        'ac': train_actor_critic_MNIST,

        # 'test': test_policy_MNIST,
        'deterministic': test_deterministic_MNIST,
        'stochastic': test_stochastic_MNIST,
        'random_drop': test_random_drop_MNIST,

        'new_train': new_train_MNIST,
    })
