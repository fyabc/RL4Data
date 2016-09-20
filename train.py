#! /usr/bin/python

from __future__ import print_function, unicode_literals

import sys

import numpy as np

from config import Config, ParamConfig
from utils import *
from DeepResidualLearning_CIFAR10 import CNN
from policyNetwork import PolicyNetwork

__author__ = 'fyabc'


def train_raw():
    # Some configures
    epoch_per_episode = ParamConfig['epoch_per_episode']

    # Create neural network model
    cnn = CNN()

    # Load the dataset
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_cifar10_data(load_cifar10_data())
    train_small_size = ParamConfig['train_epoch_size']
    x_train_small, y_train_small = get_small_train_data(x_train, y_train, train_small_size)

    message('Training data size:', y_train_small.shape[0])
    message('Validation data size:', y_validate.shape[0])
    message('Test data size:', y_test.shape[0])

    # Train the network
    batch_size = ParamConfig['train_batch_size']

    if ParamConfig['warm_start']:
        cnn.load_model(Config['model_file'])
    else:
        cnn.reset_all_parameters()

    for epoch in range(epoch_per_episode):
        print('[Epoch {}]'.format(epoch))
        message('[Epoch {}]'.format(epoch))

        x_train_small, y_train_small = shuffle_data(x_train_small, y_train_small)

        train_err = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(x_train_small, y_train_small, batch_size, shuffle=True, augment=True):
            inputs, targets = batch

            train_err += cnn.train_function(inputs, targets)
            train_batches += 1

        validate_err, validate_acc, validate_batches = cnn.validate_or_test(x_validate, y_validate)
        validate_acc /= validate_batches

        print("Epoch {} of {} took {:.3f}s".format(epoch, epoch_per_episode, time.time() - start_time))
        print('Training Loss:', train_err / train_batches)
        print('Validate Loss:', validate_err / validate_batches)
        print('#Validate accuracy:', validate_acc)

        if (epoch + 1) in (41, 61):
            cnn.update_learning_rate()

    if Config['save_model'] and validate_acc >= 0.35:
        message('Saving CNN model warm start... ', end='')
        cnn.save_model()
        message('done')

    cnn.test(x_test, y_test)


def train_policy():
    # Some configures
    num_episodes = ParamConfig['num_epochs']

    input_size = CNN.get_policy_input_size()
    print('Input size of policy network:', input_size)

    epoch_per_episode = ParamConfig['epoch_per_episode']

    # Create the policy network
    policy = PolicyNetwork(input_size=input_size)

    # Create neural network model
    cnn = CNN()

    # Load the dataset
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_cifar10_data(load_cifar10_data())
    train_small_size = ParamConfig['train_epoch_size']
    x_train_small, y_train_small = get_small_train_data(x_train, y_train, train_small_size)

    message('Training data size:', y_train_small.shape[0])
    message('Validation data size:', y_validate.shape[0])
    message('Test data size:', y_test.shape[0])

    # Train the network
    batch_size = ParamConfig['train_batch_size']

    for episode in range(1, num_episodes + 1):
        print('[Episode {}]'.format(episode))
        message('[Episode {}]'.format(episode))

        if ParamConfig['warm_start']:
            cnn.load_model(Config['model_file'])
        else:
            cnn.reset_all_parameters()
        cnn.reset_learning_rate()

        for epoch in range(epoch_per_episode):
            print('[Epoch {}]'.format(epoch))
            message('[Epoch {}]'.format(epoch))

            x_train_small, y_train_small = shuffle_data(x_train_small, y_train_small)

            total_accepted_cases = 0
            distribution = np.zeros((10,), dtype='int32')

            policy.start_new_epoch()

            train_err = 0
            train_batches = 0
            start_time = time.time()

            for batch in iterate_minibatches(x_train_small, y_train_small, batch_size, shuffle=True, augment=True):
                inputs, targets = batch

                probability = cnn.get_policy_input(inputs, targets, epoch)
                actions = policy.take_action(probability)

                # get masked inputs and targets
                inputs = inputs[actions]
                targets = targets[actions]

                total_accepted_cases += len(inputs)

                # add immediate reward
                if ParamConfig['immediate_reward']:
                    x_validate_small, y_validate_small = get_small_train_data(
                            x_validate, y_validate, ParamConfig['immediate_reward_sample_size'])
                    _, validate_acc, validate_batches = cnn.validate_or_test(x_validate_small, y_validate_small)
                    validate_acc /= validate_batches
                    policy.reward_buffer[-1].append(validate_acc)

                # print label distributions
                if ParamConfig['print_label_distribution']:
                    for target in targets:
                        distribution[target] += 1

                train_err += cnn.train_function(inputs, targets)
                train_batches += 1

            validate_err, validate_acc, validate_batches = cnn.validate_or_test(x_validate, y_validate)
            validate_acc /= validate_batches

            print("Epoch {} of {} took {:.3f}s".format(epoch, epoch_per_episode, time.time() - start_time))
            print('Training Loss:', train_err / train_batches)
            print('Validate Loss:', validate_err / validate_batches)
            print('#Validate accuracy:', validate_acc)

            message('Number of accepted cases {} of {} cases'.format(total_accepted_cases, train_small_size))
            message('Label distribution:', distribution)

            if (epoch + 1) in (41, 61):
                cnn.update_learning_rate()

        cnn.test(x_test, y_test)

        validate_err, validate_acc, validate_batches = cnn.validate_or_test(x_validate, y_validate)
        validate_acc /= validate_batches

        policy.update(validate_acc)

        if Config['policy_save_freq'] > 0 and episode % Config['policy_save_freq'] == 0:
            policy.save_policy()

        if episode % ParamConfig['policy_learning_rate_discount_freq'] == 0:
            policy.discount_learning_rate()


def train_cnn_deterministic():
    # Some configures
    curriculum = ParamConfig['curriculum']

    input_size = CNN.get_policy_input_size()
    print('Input size of policy network:', input_size)

    epoch_per_episode = ParamConfig['epoch_per_episode']

    # Load the dataset and get small training data
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_cifar10_data(load_cifar10_data())
    x_train_small, y_train_small = get_small_train_data(x_train, y_train)

    message('Training data size:', y_train_small.shape[0])
    message('Validation data size:', y_validate.shape[0])
    message('Test data size:', y_test.shape[0])

    # Create the policy network and ResNet model
    policy = PolicyNetwork(input_size=input_size)
    policy.load_policy()
    message('$    w = {}\n'
            '$    b = {}'
            .format(policy.W.get_value(), policy.b.get_value()))

    cnn = CNN()
    if ParamConfig['warm_start']:
        cnn.load_model()

    # Train the network
    batch_size = ParamConfig['train_batch_size']

    for epoch in range(epoch_per_episode):
        print('[Epoch {}]'.format(epoch))
        message('[Epoch {}]'.format(epoch))

        if not curriculum:
            x_train_small, y_train_small = shuffle_data(x_train_small, y_train_small)
        else:
            start_time = time.time()
            all_probabilities = [cnn.get_policy_input(inputs, targets, epoch) for inputs, targets in
                                 iterate_minibatches(x_train_small, y_train_small, batch_size,
                                 shuffle=False, augment=True)]
            all_alphas = np.concatenate([
                np.asarray([policy.output_function(prob) for prob in probability], dtype=fX)
                for probability in all_probabilities
            ], axis=0)
            sorted_idx_alpha = sorted(enumerate(all_alphas), key=lambda e: -e[1])
            indices = np.asarray([elem[0] for elem in sorted_idx_alpha], dtype='int64')
            x_train_small = x_train_small[indices]
            y_train_small = y_train_small[indices]

            print('Curriculum took {:.3f}s'.format(time.time() - start_time))
            print('Length of indices:', len(indices))
            print('Shape of x and y:', x_train_small.shape, y_train_small.shape)
            print('Idx_alpha:', np.asarray(sorted_idx_alpha))

        train_err = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(x_train_small, y_train_small, batch_size,
                                         shuffle=not curriculum, augment=True):
            inputs, targets = batch

            if not curriculum:
                probability = cnn.get_policy_input(inputs, targets, epoch)
                alpha = np.asarray([policy.output_function(prob) for prob in probability], dtype=fX)

                alpha /= np.sum(alpha)

                train_err += cnn.alpha_train_function(inputs, targets, alpha)
            else:
                train_err += cnn.train_function(inputs, targets)
            train_batches += 1

        if (epoch + 1) in (41, 61):
            cnn.update_learning_rate()

        validate_err, validate_acc, validate_batches = cnn.validate_or_test(x_validate, y_validate)
        validate_acc /= validate_batches

        print("Epoch {} of {} took {:.3f}s".format(epoch, epoch_per_episode, time.time() - start_time))
        print('Training Loss:', train_err / train_batches)
        print('Validate Loss:', validate_err / validate_batches)
        print('#Validate accuracy:', validate_acc)

        cnn.test(x_test, y_test)


def train_cnn_stochastic():
    # Some configures
    random_drop = ParamConfig['random_drop']

    input_size = CNN.get_policy_input_size()
    print('Input size of policy network:', input_size)

    epoch_per_episode = ParamConfig['epoch_per_episode']

    # Load the dataset and get small training data
    data = load_cifar10_data()
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_cifar10_data(data)

    train_small_size = ParamConfig['train_epoch_size']
    x_train_small, y_train_small = get_small_train_data(x_train, y_train)

    message('Training data size:', y_train_small.shape[0])
    message('Validation data size:', y_validate.shape[0])
    message('Test data size:', y_test.shape[0])

    # Create the policy network and ResNet model
    policy = PolicyNetwork(input_size=input_size)
    policy.load_policy()
    message('$    w = {}\n'
            '$    b = {}'
            .format(policy.W.get_value(), policy.b.get_value()))

    cnn = CNN()
    if ParamConfig['warm_start']:
        cnn.load_model()

    # load random drop numbers
    if random_drop:
        random_drop_numbers = map(lambda l: int(l.strip()), list(open(ParamConfig['random_drop_number_file'], 'r')))

    # Train the network
    batch_size = ParamConfig['train_batch_size']

    for epoch in range(epoch_per_episode):
        print('[Epoch {}]'.format(epoch))
        message('[Epoch {}]'.format(epoch))

        x_train_small, y_train_small = shuffle_data(x_train_small, y_train_small)

        train_err = 0
        train_batches = 0
        total_accepted_cases = 0
        start_time = time.time()

        for batch in iterate_minibatches(x_train_small, y_train_small, batch_size,
                                         shuffle=True, augment=True):
            inputs, targets = batch

            if random_drop:
                actions = np.random.binomial(1, 1 - float(random_drop_numbers[epoch]) / train_small_size,
                                             targets.shape).astype(bool)
            else:
                probability = cnn.get_policy_input(inputs, targets, epoch)
                actions = policy.take_action(probability)

            # get masked inputs and targets
            inputs = inputs[actions]
            targets = targets[actions]

            total_accepted_cases += len(inputs)

            train_err += cnn.train_function(inputs, targets)
            train_batches += 1

        if (epoch + 1) in (41, 61):
            cnn.update_learning_rate()

        validate_err, validate_acc, validate_batches = cnn.validate_or_test(x_validate, y_validate)
        validate_acc /= validate_batches

        print("Epoch {} of {} took {:.3f}s".format(epoch, epoch_per_episode, time.time() - start_time))
        print('Training Loss:', train_err / train_batches)
        print('Validate Loss:', validate_err / validate_batches)
        print('#Validate accuracy:', validate_acc)
        print('Number of accepted cases: {} of {} total'.format(total_accepted_cases, x_train_small.shape[0]))

        cnn.test(x_test, y_test)


if __name__ == '__main__':
    process_before_train(ParamConfig)

    if Config['train_type'] == 'raw':
        train_raw()
    elif Config['train_type'] == 'policy':
        train_policy()
    elif Config['train_type'] == 'deterministic':
        train_cnn_deterministic()
    elif Config['train_type'] == 'stochastic':
        train_cnn_stochastic()
    else:
        raise Exception('Unknown train type')
