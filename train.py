#! /usr/bin/python

from __future__ import print_function, unicode_literals

import sys

import numpy as np

from config import Config, ParamConfig
from utils import *
from DeepResidualLearning_CIFAR10 import CNN
from policyNetwork import PolicyNetwork

__author__ = 'fyabc'


def main():
    # Some configures
    use_policy = ParamConfig['use_policy']

    n = ParamConfig['n']
    num_episodes = ParamConfig['num_epochs']

    input_size = CNN.get_policy_input_size()
    print('Input size of policy network:', input_size)

    epoch_per_episode = ParamConfig['epoch_per_episode']

    # Create the policy network
    policy = PolicyNetwork(
            input_size=input_size,
    )

    # Create neural network model
    cnn = CNN(n)

    # Load the dataset
    data = load_cifar10_data()
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_data(data)

    train_small_size = ParamConfig['train_epoch_size']

    x_train_small, y_train_small = get_small_train_data(x_train, y_train)

    message('Training data size:', y_train_small.shape[0])
    message('Validation data size:', y_validate.shape[0])
    message('Test data size:', y_test.shape[0])

    # Train the network
    batch_size = ParamConfig['train_batch_size']

    for episode in range(1, num_episodes + 1):
        print('[Episode {}]'.format(episode))
        message('[Episode {}]'.format(episode))

        x_train_small, y_train_small = shuffle_data(x_train_small, y_train_small)

        if ParamConfig['warm_start']:
            cnn.load_model()
        else:
            cnn.reset_all_parameters()

        for epoch in range(epoch_per_episode):
            print('[Epoch {}]'.format(epoch))
            message('[Epoch {}]'.format(epoch))

            total_accepted_cases = 0
            distribution = np.zeros((10,), dtype='int32')

            for batch in iterate_minibatches(x_train_small, y_train_small, batch_size, shuffle=True, augment=True):
                inputs, targets = batch

                if use_policy:
                    probability = cnn.get_policy_input(inputs, targets)

                    actions = policy.take_action(probability)

                    # get masked inputs and targets
                    inputs = inputs[actions]
                    targets = targets[actions]

                    total_accepted_cases += len(inputs)

                # print label distributions
                if ParamConfig['print_label_distribution']:
                    for target in targets:
                        distribution[target] += 1

                train_err = cnn.train_function(inputs, targets)
                # print('Training error:', train_err / batch_size)

            validate_err, validate_acc, validate_batches = cnn.validate_or_test(x_validate, y_validate)
            validate_acc /= validate_batches

            print('Validate Loss:', validate_err / validate_batches)
            print('#Validate accuracy:', validate_acc)
            message('Number of accepted cases {} of {} cases'.format(total_accepted_cases, train_small_size))
            message('Label distribution:', distribution)

            if use_policy:
                # # get validation probabilities
                # probability = cnn.get_policy_input(x_validate, y_validate)

                policy.update(validate_acc)
                # policy.update_and_validate(validate_acc, probability)

        if use_policy:
            if episode % ParamConfig['policy_learning_rate_discount_freq'] == 0:
                policy.discount_learning_rate()

            if Config['policy_save_freq'] > 0 and episode % Config['policy_save_freq'] == 0:
                policy.save_policy()

        if Config['save_model'] and not use_policy and validate_acc >= 0.35:
            message('Saving CNN model warm start... ', end='')
            cnn.save_model()
            message('done')
            break

    cnn.test(x_test, y_test)


def train_deterministic():
    # Some configures
    use_policy = ParamConfig['use_policy']

    n = ParamConfig['n']
    num_episodes = ParamConfig['num_epochs']

    input_size = CNN.get_policy_input_size()
    print('Input size of policy network:', input_size)

    epoch_per_episode = ParamConfig['epoch_per_episode']

    # Create the policy network
    policy = PolicyNetwork(
            input_size=input_size,
    )

    # Create neural network model
    cnn = CNN(n)

    # Load the dataset
    data = load_cifar10_data()
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_data(data)

    train_small_size = ParamConfig['train_epoch_size']

    x_train_small, y_train_small = get_small_train_data(x_train, y_train)

    message('Training data size:', y_train_small.shape[0])
    # message('Validation data size:', y_validate.shape[0])
    message('Test data size:', y_test.shape[0])

    # Train the network
    batch_size = ParamConfig['train_batch_size']

    policy.load_policy()

    message('$    w = {}\n'
            '$    b = {}'
            .format(policy.W.get_value(), policy.b.get_value()))

    for episode in range(1, num_episodes + 1):
        print('[Episode {}]'.format(episode))
        message('[Episode {}]'.format(episode))

        cnn.load_model()

        x_train_small, y_train_small = shuffle_data(x_train_small, y_train_small)

        for epoch in range(epoch_per_episode):
            print('[Epoch {}]'.format(epoch))
            message('[Epoch {}]'.format(epoch))

            for batch in iterate_minibatches(x_train_small, y_train_small, batch_size, shuffle=True, augment=True):
                inputs, targets = batch
                probability = cnn.get_policy_input(inputs, targets)

                alpha = np.asarray(map(lambda prob: policy.output_function(prob), probability), dtype=fX).flatten()
                alpha /= np.sum(alpha)

                train_err = cnn.alpha_train_function(inputs, targets, alpha)
                # print('Training error:', train_err / batch_size)

            validate_err, validate_acc, validate_batches = cnn.validate_or_test(x_validate, y_validate)
            validate_acc /= validate_batches

            print('Validate Loss:', validate_err / validate_batches)
            print('#Validate accuracy:', validate_acc)

            # TODO
            # if use_policy:
            #     policy.update_deterministic(validate_acc)

        if episode % ParamConfig['policy_learning_rate_discount_freq'] == 0:
            policy.discount_learning_rate()


if __name__ == '__main__':
    import pprint

    argc = len(sys.argv)

    if '-h' in sys.argv or '--help' in sys.argv:
        print('Usage: add properties just like this:\n'
              '    add_label_prob=False\n'
              '    #policy_save_freq=10\n'
              '\n'
              'properties starts with # are in Config, other properties are in ParamConfig.')

    args_dict, param_args_dict = simple_parse_args(sys.argv)
    Config.update(args_dict)
    ParamConfig.update(param_args_dict)

    message('The configures and hyperparameters are:')
    pprint.pprint(ParamConfig, stream=sys.stderr)

    if ParamConfig['train_deterministic']:
        train_deterministic()
    else:
        main()
