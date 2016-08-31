#! /usr/bin/python

from __future__ import print_function, unicode_literals

import sys
import random

import numpy as np

from config import Config, ParamConfig
from utils import load_cifar10_data, iterate_minibatches, message, simple_parse_args, fX
from DeepResidualLearning_CIFAR10 import CNN
from policyNetwork import PolicyNetwork

__author__ = 'fyabc'


def main():
    # Some configures
    use_policy = False

    n = ParamConfig['n']
    num_episodes = ParamConfig['num_epochs']

    input_size = CNN.get_policy_input_size()
    print('Input size of policy network:', input_size)

    epoch_per_episode = 2

    # Create the policy network
    policy = PolicyNetwork(
        input_size=input_size,
    )

    # Create neural network model
    cnn = CNN(n)

    # Load the dataset
    data = load_cifar10_data()
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    train_size = x_train.shape[0]
    train_episode_size = ParamConfig['train_epoch_size']

    # Train the network
    batch_size = ParamConfig['train_batch_size']

    cnn.load_model()

    for episode in range(1, num_episodes + 1):
        print('[Episode {}]'.format(episode))
        message('[Episode {}]'.format(episode))

        if ParamConfig['warm_start']:
            cnn.load_model()
        else:
            cnn.reset_all_parameters()

        for epoch in range(epoch_per_episode):
            print('[Epoch {}]'.format(epoch))
            message('[Epoch {}]'.format(epoch))

            # Use small dataset to check the code
            sampled_indices = random.sample(range(train_size), train_episode_size)
            x_train_epoch = x_train[sampled_indices]
            y_train_epoch = y_train[sampled_indices]

            total_accepted_cases = 0

            for batch in iterate_minibatches(x_train_epoch, y_train_epoch, batch_size, shuffle=True, augment=True):
                inputs, targets = batch

                if use_policy:
                    probability = cnn.get_policy_input(inputs, targets)

                    actions = policy.take_action(probability)

                    # get masked inputs and targets
                    inputs = inputs[actions]
                    targets = targets[actions]

                    total_accepted_cases += len(inputs)

                train_err = cnn.train_function(inputs, targets)
                # print('Training error:', train_err)

            validate_err, validate_acc, validate_batches = cnn.validate_or_test(x_test, y_test)
            validate_acc /= validate_batches

            print('Validate Loss:', validate_err / validate_batches)
            print('#Validate accuracy:', validate_acc)
            message('Number of accepted cases {} of {} cases'.format(total_accepted_cases, train_episode_size))

            if use_policy:
                # get validation probabilities
                probability = cnn.get_policy_input(x_test, y_test)

                # policy.update(validate_acc)
                policy.update_and_validate(validate_acc, probability)

                if episode % ParamConfig['policy_learning_rate_discount_freq'] == 0:
                    policy.discount_learning_rate()

    cnn.test(x_test, y_test)


if __name__ == '__main__':
    import pprint

    argc = len(sys.argv)

    if '-h' in sys.argv or '--help' in sys.argv:
        print('Usage: add properties just like this:\n'
              '    add_label_prob=False')

    ParamConfig.update(simple_parse_args(sys.argv))

    message('The configures and hyperparameters are:')
    pprint.pprint(ParamConfig, stream=sys.stderr)

    main()
