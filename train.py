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
    use_policy = True

    n = ParamConfig['n']
    num_epochs = ParamConfig['num_epochs']

    input_size = ParamConfig['cnn_output_size']
    if ParamConfig['add_label_input']:
        input_size += 1

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
    train_epoch_size = ParamConfig['train_epoch_size']

    # Train the network
    batch_size = ParamConfig['train_batch_size']

    for epoch in range(1, num_epochs + 1):
        if use_policy:
            if ParamConfig['warm_start']:
                cnn.load_model()
            else:
                cnn.reset_all_parameters()

        # Use small dataset to check the code
        sampled_indices = random.sample(range(train_size), train_epoch_size)
        x_train_epoch = x_train[sampled_indices]
        y_train_epoch = y_train[sampled_indices]

        for batch in iterate_minibatches(x_train_epoch, y_train_epoch, batch_size, shuffle=True, augment=True):
            inputs, targets = batch

            if use_policy:
                probability = cnn.probs_function(inputs)

                if ParamConfig['add_label_input']:
                    label_inputs = np.zeros(shape=(batch_size, 1), dtype=fX)
                    for i in range(batch_size):
                        label_inputs[i, 0] = probability[i, targets[i]]
                    probability = np.hstack([probability, label_inputs])

                actions = policy.take_action(probability)

                # get masked inputs and targets
                inputs = inputs[actions]
                targets = targets[actions]

                # print('Number of accepted cases:', len(inputs))

            train_err = cnn.train_function(inputs, targets)
            # print('Training error:', train_err)

        _, validate_acc, validate_batches = cnn.validate_or_test(x_test, y_test)
        validate_acc /= validate_batches

        print('#Validate accuracy:', validate_acc)

        if use_policy:
            # get validation probabilities
            probability = cnn.probs_function(x_test)

            if ParamConfig['add_label_input']:
                label_inputs = np.zeros(shape=(y_test.shape[0], 1), dtype=fX)
                for i in range(batch_size):
                    label_inputs[i, 0] = probability[i, y_test[i]]
                probability = np.hstack([probability, label_inputs])

            # policy.update(validate_acc)
            policy.update_and_validate(validate_acc, probability)

            if epoch % ParamConfig['policy_learning_rate_discount_freq'] == 0:
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
