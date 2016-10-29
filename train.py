#! /usr/bin/python

from __future__ import print_function, unicode_literals

import sys
import heapq
import traceback
from collections import deque

import numpy as np

from config import Config, CifarConfig, PolicyConfig
from utils import *
from model_CIFAR10 import CIFARModel, VaniliaCNNModel
from policyNetwork import PolicyNetwork
from criticNetwork import CriticNetwork

__author__ = 'fyabc'


def epoch_message(model, x_train, y_train, x_validate, y_validate, x_test, y_test,
                  history_accuracy, history_train_loss,
                  epoch, start_time, train_batches, total_accepted_cases):
    # Get training loss
    train_loss = model.get_training_loss(x_train, y_train)

    # Get validation loss and accuracy
    validate_loss, validate_acc, validate_batches = model.validate_or_test(x_validate, y_validate)
    validate_loss /= validate_batches
    validate_acc /= validate_batches
    history_accuracy.append(validate_acc)

    # Get test loss and accuracy
    test_loss, test_acc, test_batches = model.validate_or_test(x_test, y_test)
    test_loss /= test_batches
    test_acc /= test_batches

    message("Epoch {} of {} took {:.3f}s".format(epoch, CifarConfig['epoch_per_episode'], time.time() - start_time))
    message('Training Loss:', train_loss)
    message('History Training Loss:', history_train_loss / train_batches)
    message('Validate Loss:', validate_loss)
    message('#Validate accuracy:', validate_acc)
    message('Test Loss:', test_loss),
    message('#Test accuracy:', test_acc)
    message('Number of accepted cases: {} of {} total'.format(total_accepted_cases, x_train.shape[0]))


def train_raw_CIFAR10():
    model_name = eval(CifarConfig['model_name'])
    # Create neural network model
    model = model_name()
    # model = VaniliaCNNModel()

    # Load the dataset
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_cifar10_data(load_cifar10_data())

    message('Training data size:', y_train.shape[0])
    message('Validation data size:', y_validate.shape[0])
    message('Test data size:', y_test.shape[0])

    # Train the network
    if CifarConfig['warm_start']:
        model.load_model(Config['model_file'])
    else:
        model.reset_parameters()

    # Iteration (number of batches)
    iteration = 0

    for epoch in range(CifarConfig['epoch_per_episode']):
        print('[Epoch {}]'.format(epoch))
        message('[Epoch {}]'.format(epoch))

        total_accepted_cases = 0
        history_train_loss = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(x_train, y_train, CifarConfig['train_batch_size'], shuffle=True, augment=True):
            iteration += 1

            inputs, targets = batch

            total_accepted_cases += len(inputs)

            part_train_err = model.f_train(inputs, targets)

            if iteration % CifarConfig['display_freq'] == 0:
                message('Train error of iteration {} is {}'.format(iteration, part_train_err))

            history_train_loss += part_train_err
            train_batches += 1

        epoch_message(model, x_train, y_train, x_validate, y_validate, x_test, y_test,
                      [], history_train_loss,
                      epoch, start_time, train_batches, total_accepted_cases)

        if model_name == CIFARModel:
            if (epoch + 1) in (41, 61):
                model.update_learning_rate()

    if Config['save_model']:
        message('Saving CNN model warm start... ', end='')
        model.save_model()
        message('done')

    model.test(x_test, y_test)


def train_SPL_CIFAR10():
    model_name = eval(CifarConfig['model_name'])
    # Create neural network model
    model = model_name()
    # model = VaniliaCNNModel()

    # Load the dataset
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_cifar10_data(load_cifar10_data())

    message('Training data size:', y_train.shape[0])
    message('Validation data size:', y_validate.shape[0])
    message('Test data size:', y_test.shape[0])

    # Self-paced learning iterate on data cases
    total_iteration_number = CifarConfig['epoch_per_episode'] * len(x_train) // model.train_batch_size

    # Get the cost threshold \lambda.
    def cost_threshold(iteration):
        return 1 + (model.train_batch_size - 1) * iteration / total_iteration_number

    # Data buffer
    spl_buffer = deque()

    # When collect such number of cases, update them.
    update_maxlen = model.train_batch_size

    # # Self-paced learning setting end

    # Train the network
    if CifarConfig['warm_start']:
        model.load_model(Config['model_file'])
    else:
        model.reset_parameters()

    # Iteration (number of batches)
    iteration = 0

    for epoch in range(CifarConfig['epoch_per_episode']):
        print('[Epoch {}]'.format(epoch))
        message('[Epoch {}]'.format(epoch))

        total_accepted_cases = 0
        history_train_loss = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(x_train, y_train, CifarConfig['train_batch_size'],
                                         shuffle=True, augment=True, return_indices=True):
            iteration += 1

            inputs, targets, indices = batch

            selected_number = cost_threshold(iteration)

            cost_list = model.f_cost_list_without_decay(inputs, targets)
            label_cost_lists = [cost_list[targets == label] for label in range(10)]

            for i, label_cost_list in enumerate(label_cost_lists):
                if label_cost_list.size != 0:
                    threshold = heapq.nsmallest(selected_number, label_cost_list)[-1]
                    for j in range(len(targets)):
                        if targets[j] == i and cost_list[j] <= threshold:
                            spl_buffer.append(indices[j])

            if len(spl_buffer) >= update_maxlen:
                # message('SPL buffer full, update...', end='')

                update_batch_index = [spl_buffer.popleft() for _ in range(update_maxlen)]
                # Get masked inputs and targets
                inputs_selected = x_train[update_batch_index]
                targets_selected = y_train[update_batch_index]

                total_accepted_cases += len(inputs_selected)

                part_train_err = model.f_train(inputs_selected, targets_selected)

                # message('done')
            else:
                part_train_err = None

            # # In SPL, display after each update
            # if iteration % CifarConfig['display_freq'] == 0:
            if part_train_err is not None:
                message('Train error of iteration {} is {}'.format(iteration, part_train_err))
                history_train_loss += part_train_err
            train_batches += 1

        epoch_message(model, x_train, y_train, x_validate, y_validate, x_test, y_test,
                      [], history_train_loss,
                      epoch, start_time, train_batches, total_accepted_cases)

        if model_name == CIFARModel:
            if (epoch + 1) in (41, 61):
                model.update_learning_rate()

    model.test(x_test, y_test)


def train_policy_CIFAR10():
    model_name = eval(CifarConfig['model_name'])
    # Create neural network model
    model = model_name()
    # model = VaniliaCNNModel()

    # Create the policy network
    input_size = CIFARModel.get_policy_input_size()
    print('Input size of policy network:', input_size)
    policy = PolicyNetwork(input_size=input_size)

    # Load the dataset
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_cifar10_data(load_cifar10_data())
    train_small_size = CifarConfig['train_small_size']
    x_train_small, y_train_small = get_part_data(x_train, y_train, train_small_size)

    message('Training data size:', y_train_small.shape[0])
    message('Validation data size:', y_validate.shape[0])
    message('Test data size:', y_test.shape[0])

    # Train the network
    for episode in range(1, PolicyConfig['num_episodes'] + 1):
        print('[Episode {}]'.format(episode))
        message('[Episode {}]'.format(episode))

        if CifarConfig['warm_start']:
            model.load_model(Config['model_file'])
        else:
            model.reset_parameters()
        model.reset_learning_rate()

        history_accuracy = []

        # Speed reward
        first_over_index = None

        for epoch in range(CifarConfig['epoch_per_episode']):
            print('[Epoch {}]'.format(epoch))
            message('[Epoch {}]'.format(epoch))

            policy.start_new_epoch()

            total_accepted_cases = 0
            history_train_loss = 0
            train_batches = 0
            start_time = time.time()

            for batch in iterate_minibatches(x_train_small, y_train_small, model.train_batch_size,
                                             shuffle=True, augment=True):
                inputs, targets = batch

                probability = model.get_policy_input(inputs, targets, epoch, history_accuracy)
                actions = policy.take_action(probability)

                # get masked inputs and targets
                inputs = inputs[actions]
                targets = targets[actions]

                total_accepted_cases += len(inputs)

                history_train_loss += model.f_train(inputs, targets)
                train_batches += 1

            validate_loss, validate_acc, validate_batches = model.validate_or_test(x_validate, y_validate)
            validate_acc /= validate_batches

            history_accuracy.append(validate_acc)

            # Check speed rewards
            if first_over_index is None and validate_acc >= PolicyConfig['speed_reward_threshold']:
                first_over_index = epoch

            if model_name == CIFARModel:
                if (epoch + 1) in (41, 61):
                    model.update_learning_rate()

            # add immediate reward
            if PolicyConfig['immediate_reward']:
                x_validate_small, y_validate_small = get_part_data(
                    x_validate, y_validate, PolicyConfig['immediate_reward_sample_size'])
                _, validate_acc, validate_batches = model.validate_or_test(x_validate_small, y_validate_small)
                validate_acc /= validate_batches
                policy.reward_buffer.append(validate_acc)

            epoch_message(model, x_train, y_train, x_validate, y_validate, x_test, y_test,
                          history_accuracy, history_train_loss,
                          epoch, start_time, train_batches, total_accepted_cases)

        model.test(x_test, y_test)

        validate_loss, validate_acc, validate_batches = model.validate_or_test(x_validate, y_validate)
        validate_acc /= validate_batches

        # Updating policy
        if PolicyConfig['speed_reward']:
            if first_over_index is None:
                first_over_index = CifarConfig['epoch_per_episode']
            terminal_reward = floatX(first_over_index) / CifarConfig['epoch_per_episode']
            policy.update(-np.log(terminal_reward))
        else:
            policy.update(validate_acc)

        if Config['policy_save_freq'] > 0 and episode % Config['policy_save_freq'] == 0:
            policy.save_policy()


def train_actor_critic_CIFAR10():
    model_name = eval(CifarConfig['model_name'])
    # Create neural network model
    model = model_name()
    # model = VaniliaCNNModel()

    # Create the actor network
    input_size = CIFARModel.get_policy_input_size()
    print('Input size of actor network:', input_size)
    actor = PolicyNetwork(input_size=input_size)
    critic = CriticNetwork(feature_size=input_size, batch_size=model.train_batch_size)

    # Load the dataset
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_cifar10_data(load_cifar10_data())
    train_small_size = CifarConfig['train_small_size']
    x_train_small, y_train_small = get_part_data(x_train, y_train, train_small_size)

    message('Training data size:', y_train_small.shape[0])
    message('Validation data size:', y_validate.shape[0])
    message('Test data size:', y_test.shape[0])

    # Train the network
    for episode in range(1, PolicyConfig['num_episodes'] + 1):
        print('[Episode {}]'.format(episode))
        message('[Episode {}]'.format(episode))

        if CifarConfig['warm_start']:
            model.load_model(Config['model_file'])
        else:
            model.reset_parameters()
        model.reset_learning_rate()

        history_accuracy = []

        # Iteration (number of batches)
        iteration = 0

        for epoch in range(CifarConfig['epoch_per_episode']):
            print('[Epoch {}]'.format(epoch))
            message('[Epoch {}]'.format(epoch))

            actor.start_new_epoch()

            total_accepted_cases = 0
            history_train_loss = 0
            train_batches = 0
            start_time = time.time()

            for batch in iterate_minibatches(x_train_small, y_train_small, model.train_batch_size,
                                             shuffle=True, augment=True):
                iteration += 1

                inputs, targets = batch

                probability = model.get_policy_input(inputs, targets, epoch, history_accuracy)
                actions = actor.take_action(probability)

                # get masked inputs and targets
                inputs_selected = inputs[actions]
                targets_selected = targets[actions]

                total_accepted_cases += len(inputs_selected)

                # Update the CIFAR10 network with selected data
                part_train_err = model.f_train(inputs_selected, targets_selected)
                history_train_loss += part_train_err
                train_batches += 1

                # Get immediate reward
                if PolicyConfig['cost_gap_AC_reward']:
                    cost_old = part_train_err
                    cost_new = model.f_cost_without_decay(inputs, targets)
                    imm_reward = cost_old - cost_new
                else:
                    valid_part_x, valid_part_y = get_part_data(
                        np.asarray(x_validate), np.asarray(y_validate), PolicyConfig['immediate_reward_sample_size'])
                    _, valid_err, validate_batches = model.validate_or_test(valid_part_x, valid_part_y)
                    imm_reward = valid_err / validate_batches

                # Get new state, new actions, and compute new Q value
                probability_new = model.get_policy_input(inputs, targets, epoch, history_accuracy)
                actions_new = actor.take_action(probability_new, log_replay=False)

                Q_value_new = critic.Q_function(state=probability_new, action=actions_new)
                if epoch < CifarConfig['epoch_per_episode'] - 1:
                    label = PolicyConfig['actor_gamma'] * Q_value_new + imm_reward
                else:
                    label = imm_reward

                # Update the critic Q network
                Q_loss = critic.update(probability, actions, floatX(label))

                # Update actor network
                actor_loss = actor.update_raw(probability, actions,
                                              np.full(actions.shape, label, dtype=probability.dtype))

                if iteration % CifarConfig['display_freq'] == 0:
                    message('Epoch {}\tIteration {}\tCost {}\tCritic loss {}\tActor loss {}'
                            .format(epoch, iteration, part_train_err, Q_loss, actor_loss))

            if model_name == CIFARModel:
                if (epoch + 1) in (41, 61):
                    model.update_learning_rate()

            # add immediate reward
            if PolicyConfig['immediate_reward']:
                x_validate_small, y_validate_small = get_part_data(
                    x_validate, y_validate, PolicyConfig['immediate_reward_sample_size'])
                _, validate_acc, validate_batches = model.validate_or_test(x_validate_small, y_validate_small)
                validate_acc /= validate_batches
                actor.reward_buffer.append(validate_acc)

            epoch_message(model, x_train, y_train, x_validate, y_validate, x_test, y_test,
                          history_accuracy, history_train_loss,
                          epoch, start_time, train_batches, total_accepted_cases)

        model.test(x_test, y_test)

        validate_loss, validate_acc, validate_batches = model.validate_or_test(x_validate, y_validate)
        validate_acc /= validate_batches

        actor.update(validate_acc)

        if Config['policy_save_freq'] > 0 and episode % Config['policy_save_freq'] == 0:
            actor.save_policy()

        if episode % PolicyConfig['policy_learning_rate_discount_freq'] == 0:
            actor.discount_learning_rate()


def test_policy_CIFAR10():
    model_name = eval(CifarConfig['model_name'])
    # Create neural network model
    model = model_name()
    # model = VaniliaCNNModel()

    input_size = CIFARModel.get_policy_input_size()
    print('Input size of policy network:', input_size)

    # Load the dataset and get small training data
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_cifar10_data(load_cifar10_data())

    message('Training data size:', y_train.shape[0])
    message('Validation data size:', y_validate.shape[0])
    message('Test data size:', y_test.shape[0])

    if CifarConfig['warm_start']:
        model.load_model()

    if Config['train_type'] == 'random_drop':
        # Random drop configure
        random_drop_numbers = map(lambda l: int(l.strip()), list(open(CifarConfig['random_drop_number_file'], 'r')))
    else:
        # Build policy
        policy = PolicyNetwork(input_size=input_size)
        policy.load_policy()
        message('$    w = {}\n'
                '$    b = {}'
                .format(policy.W.get_value(), policy.b.get_value()))

    history_accuracy = []

    # Train the network
    for epoch in range(CifarConfig['epoch_per_episode']):
        print('[Epoch {}]'.format(epoch))
        message('[Epoch {}]'.format(epoch))

        history_train_loss = 0
        train_batches = 0
        total_accepted_cases = 0
        start_time = time.time()

        for batch in iterate_minibatches(x_train, y_train, model.train_batch_size,
                                         shuffle=True, augment=True):
            inputs, targets = batch

            probability = model.get_policy_input(inputs, targets, epoch, history_accuracy)

            if Config['train_type'] == 'deterministic':
                alpha = np.asarray([policy.output_function(prob) for prob in probability], dtype=fX)

                alpha /= np.sum(alpha)

                history_train_loss += model.f_alpha_train(inputs, targets, alpha)
            else:
                if Config['train_type'] == 'stochastic':
                    actions = policy.take_action(probability, False)
                elif Config['train_type'] == 'random_drop':
                    actions = np.random.binomial(
                        1,
                        1 - float(random_drop_numbers[epoch]) / CifarConfig['train_small_size'],
                        targets.shape
                    ).astype(bool)

                # get masked inputs and targets
                inputs = inputs[actions]
                targets = targets[actions]

                history_train_loss += model.f_train(inputs, targets)

            total_accepted_cases += len(inputs)
            train_batches += 1

        if model_name == CIFARModel:
            if (epoch + 1) in (41, 61):
                model.update_learning_rate()

        epoch_message(model, x_train, y_train, x_validate, y_validate, x_test, y_test,
                      history_accuracy, history_train_loss,
                      epoch, start_time, train_batches, total_accepted_cases)

        model.test(x_test, y_test)


if __name__ == '__main__':
    process_before_train(CifarConfig)

    try:
        if Config['train_type'] == 'raw':
            train_raw_CIFAR10()
        elif Config['train_type'] == 'self_paced':
            train_SPL_CIFAR10()
        elif Config['train_type'] == 'policy':
            train_policy_CIFAR10()
        elif Config['train_type'] == 'actor_critic':
            train_actor_critic_CIFAR10()
        elif Config['train_type'] == 'deterministic':
            test_policy_CIFAR10()
        elif Config['train_type'] == 'stochastic':
            test_policy_CIFAR10()
        elif Config['train_type'] == 'random_drop':
            test_policy_CIFAR10()
        else:
            raise Exception('Unknown train type {}'.format(Config['train_type']))
    except:
        message(traceback.format_exc())
    finally:
        finalize_logging_file()
