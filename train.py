#! /usr/bin/python

from __future__ import print_function, unicode_literals

import sys

import numpy as np

from config import Config, CifarConfig, PolicyConfig
from utils import *
from model_CIFAR10 import CIFARModel
from policyNetwork import PolicyNetwork
from criticNetwork import CriticNetwork

__author__ = 'fyabc'


def train_raw_CIFAR10():
    # Create neural network model
    model = CIFARModel()

    # Load the dataset
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_cifar10_data(load_cifar10_data())
    train_small_size = CifarConfig['train_small_size']
    x_train_small, y_train_small = get_part_data(x_train, y_train, train_small_size)

    message('Training data size:', y_train_small.shape[0])
    message('Validation data size:', y_validate.shape[0])
    message('Test data size:', y_test.shape[0])

    # Train the network
    if CifarConfig['warm_start']:
        model.load_model(Config['model_file'])
    else:
        model.reset_parameters()

    for epoch in range(CifarConfig['epoch_per_episode']):
        print('[Epoch {}]'.format(epoch))
        message('[Epoch {}]'.format(epoch))

        x_train_small, y_train_small = shuffle_data(x_train_small, y_train_small)

        train_err = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(x_train_small, y_train_small,
                                         CifarConfig['train_batch_size'], shuffle=True, augment=True):
            inputs, targets = batch

            train_err += model.train_function(inputs, targets)
            train_batches += 1

        validate_err, validate_acc, validate_batches = model.validate_or_test(x_validate, y_validate)
        validate_acc /= validate_batches

        print("Epoch {} of {} took {:.3f}s".format(epoch, CifarConfig['epoch_per_episode'], time.time() - start_time))
        print('Training Loss:', train_err / train_batches)
        print('Validate Loss:', validate_err / validate_batches)
        print('#Validate accuracy:', validate_acc)

        if (epoch + 1) in (41, 61):
            model.update_learning_rate()

    if Config['save_model'] and validate_acc >= 0.35:
        message('Saving CNN model warm start... ', end='')
        model.save_model()
        message('done')

    model.test(x_test, y_test)


def train_policy_CIFAR10():
    # Create the policy network
    input_size = CIFARModel.get_policy_input_size()
    print('Input size of policy network:', input_size)
    policy = PolicyNetwork(input_size=input_size)

    # Create neural network model
    model = CIFARModel()

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

        for epoch in range(CifarConfig['epoch_per_episode']):
            print('[Epoch {}]'.format(epoch))
            message('[Epoch {}]'.format(epoch))

            x_train_small, y_train_small = shuffle_data(x_train_small, y_train_small)

            total_accepted_cases = 0
            distribution = np.zeros((10,), dtype='int32')

            policy.start_new_epoch()

            train_err = 0
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

                # print label distributions
                if CifarConfig['print_label_distribution']:
                    for target in targets:
                        distribution[target] += 1

                train_err += model.train_function(inputs, targets)
                train_batches += 1

            validate_err, validate_acc, validate_batches = model.validate_or_test(x_validate, y_validate)
            validate_acc /= validate_batches

            history_accuracy.append(validate_acc)

            if (epoch + 1) in (41, 61):
                model.update_learning_rate()

            # add immediate reward
            if PolicyConfig['immediate_reward']:
                x_validate_small, y_validate_small = get_part_data(
                    x_validate, y_validate, PolicyConfig['immediate_reward_sample_size'])
                _, validate_acc, validate_batches = model.validate_or_test(x_validate_small, y_validate_small)
                validate_acc /= validate_batches
                policy.reward_buffer.append(validate_acc)

            print("Epoch {} of {} took {:.3f}s"
                  .format(epoch, CifarConfig['epoch_per_episode'], time.time() - start_time))
            print('Training Loss:', train_err / train_batches)
            print('Validate Loss:', validate_err / validate_batches)
            print('#Validate accuracy:', validate_acc)

            message('Number of accepted cases {} of {} cases'.format(total_accepted_cases, train_small_size))
            message('Label distribution:', distribution)

        model.test(x_test, y_test)

        validate_err, validate_acc, validate_batches = model.validate_or_test(x_validate, y_validate)
        validate_acc /= validate_batches

        policy.update(validate_acc)

        if Config['policy_save_freq'] > 0 and episode % Config['policy_save_freq'] == 0:
            policy.save_policy()

        if episode % PolicyConfig['policy_learning_rate_discount_freq'] == 0:
            policy.discount_learning_rate()


def train_actor_critic_CIFAR10():
    input_size = CIFARModel.get_policy_input_size()
    print('Input size of policy network:', input_size)

    # Load the dataset and get small training data
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_cifar10_data(load_cifar10_data())
    x_train_small, y_train_small = get_part_data(x_train, y_train)

    message('Training data size:', y_train_small.shape[0])
    message('Validation data size:', y_validate.shape[0])
    message('Test data size:', y_test.shape[0])

    # Build model
    model = CIFARModel()
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

        x_train_small, y_train_small = shuffle_data(x_train_small, y_train_small)

        train_err = 0
        train_batches = 0
        total_accepted_cases = 0
        start_time = time.time()

        for batch in iterate_minibatches(x_train_small, y_train_small, model.train_batch_size,
                                         shuffle=True, augment=True):
            inputs, targets = batch

            probability = model.get_policy_input(inputs, targets, epoch, history_accuracy)

            if Config['train_type'] == 'deterministic':
                alpha = np.asarray([policy.output_function(prob) for prob in probability], dtype=fX)

                alpha /= np.sum(alpha)

                train_err += model.alpha_train_function(inputs, targets, alpha)
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

                train_err += model.train_function(inputs, targets)

            total_accepted_cases += len(inputs)
            train_batches += 1

        if (epoch + 1) in (41, 61):
            model.update_learning_rate()

        validate_err, validate_acc, validate_batches = model.validate_or_test(x_validate, y_validate)
        validate_acc /= validate_batches
        history_accuracy.append(validate_acc)

        print("Epoch {} of {} took {:.3f}s".format(epoch, CifarConfig['epoch_per_episode'], time.time() - start_time))
        print('Training Loss:', train_err / train_batches)
        print('Validate Loss:', validate_err / validate_batches)
        print('#Validate accuracy:', validate_acc)
        print('Number of accepted cases: {} of {} total'.format(total_accepted_cases, x_train_small.shape[0]))

        model.test(x_test, y_test)


def test_policy_CIFAR10():
    input_size = CIFARModel.get_policy_input_size()
    print('Input size of policy network:', input_size)

    # Load the dataset and get small training data
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_cifar10_data(load_cifar10_data())
    x_train_small, y_train_small = get_part_data(x_train, y_train)

    message('Training data size:', y_train_small.shape[0])
    message('Validation data size:', y_validate.shape[0])
    message('Test data size:', y_test.shape[0])

    # Build model
    model = CIFARModel()
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

        x_train_small, y_train_small = shuffle_data(x_train_small, y_train_small)

        train_err = 0
        train_batches = 0
        total_accepted_cases = 0
        start_time = time.time()

        for batch in iterate_minibatches(x_train_small, y_train_small, model.train_batch_size,
                                         shuffle=True, augment=True):
            inputs, targets = batch

            probability = model.get_policy_input(inputs, targets, epoch, history_accuracy)

            if Config['train_type'] == 'deterministic':
                alpha = np.asarray([policy.output_function(prob) for prob in probability], dtype=fX)

                alpha /= np.sum(alpha)

                train_err += model.alpha_train_function(inputs, targets, alpha)
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

                train_err += model.train_function(inputs, targets)

            total_accepted_cases += len(inputs)
            train_batches += 1

        if (epoch + 1) in (41, 61):
            model.update_learning_rate()

        validate_err, validate_acc, validate_batches = model.validate_or_test(x_validate, y_validate)
        validate_acc /= validate_batches
        history_accuracy.append(validate_acc)

        print("Epoch {} of {} took {:.3f}s".format(epoch, CifarConfig['epoch_per_episode'], time.time() - start_time))
        print('Training Loss:', train_err / train_batches)
        print('Validate Loss:', validate_err / validate_batches)
        print('#Validate accuracy:', validate_acc)
        print('Number of accepted cases: {} of {} total'.format(total_accepted_cases, x_train_small.shape[0]))

        model.test(x_test, y_test)


if __name__ == '__main__':
    process_before_train(CifarConfig)

    if Config['train_type'] == 'raw':
        train_raw_CIFAR10()
    elif Config['train_type'] == 'self_paced':
        train_raw_CIFAR10()
    elif Config['train_type'] == 'policy':
        train_policy_CIFAR10()
    elif Config['train_type'] == 'deterministic':
        test_policy_CIFAR10()
    elif Config['train_type'] == 'stochastic':
        test_policy_CIFAR10()
    elif Config['train_type'] == 'random_drop':
        test_policy_CIFAR10()
    else:
        raise Exception('Unknown train type {}'.format(Config['train_type']))
