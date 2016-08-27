#! /usr/bin/python

from __future__ import print_function, unicode_literals

import sys

# import lasagne

from config import Config, ParamConfig
from utils import load_cifar10_data, iterate_minibatches
from DeepResidualLearning_CIFAR10 import CNN
from policyNetwork import PolicyNetwork

__author__ = 'fyabc'


def main(n=ParamConfig['n'], num_epochs=ParamConfig['num_epochs']):
    # Create the policy network
    policy = PolicyNetwork()

    # Create neural network model
    cnn = CNN(n)

    # Load the dataset
    data = load_cifar10_data()
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    # Use small dataset to check the code
    x_train = x_train[:1280]
    y_train = y_train[:1280]

    # Train the network
    for epoch in range(num_epochs):
        for batch in iterate_minibatches(x_train, y_train, ParamConfig['train_batch_size'], shuffle=True, augment=True):
            inputs, targets = batch
            probability = cnn.probs_function(inputs)

            actions = policy.take_action(probability)

            # get masked inputs and targets
            inputs = inputs[actions]
            targets = targets[actions]
            print('Number of accepted cases:', len(inputs))

            train_err = cnn.train_function(inputs, targets)
            print('Training error:', train_err)

        _, validate_acc, validate_batches = cnn.validate_or_test(x_test, y_test)
        validate_acc /= validate_batches

        print('Validate accuracy:', validate_acc)

        policy.update(validate_acc)

    cnn.test(x_test, y_test)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a Deep Residual Learning network on cifar-10 using Lasagne.")
        print("Network architecture and training parameters are as in section 4.2 in "
              "'Deep Residual Learning for Image Recognition'.")
        print("Usage: %s [N [MODEL]]" % sys.argv[0])
        print()
        print("N: Number of stacked residual building blocks per feature map (default: 5)")
        print("MODEL: saved model file to load (for validation) (default: None)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['n'] = int(sys.argv[1])
        if len(sys.argv) > 2:
            kwargs['model'] = sys.argv[2]
        main(**kwargs)
