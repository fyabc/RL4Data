#! /usr/bin/python

from __future__ import print_function, unicode_literals

import sys

import theano.tensor as T
# import lasagne

from config import Config, ParamConfig
from utils import load_cifar10_data
from DeepResidualLearning_CIFAR10 import CNN
from policyNetwork import PolicyNetwork

__author__ = 'fyabc'


def main(n=ParamConfig['n'], num_epochs=ParamConfig['num_epochs'], model=Config['model']):
    # Load the dataset
    data = load_cifar10_data()
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    # Create neural network model
    cnn = CNN(n)
    cnn.build_train_function()

    # Create the policy network
    policy = PolicyNetwork()

    if model is None:
        # Train the network
        for epoch in range(num_epochs):
            result = cnn.train_one_epoch(x_train, y_train, x_test, y_test)

            validate_acc = result['validate_acc']
            softmax_probabilities = result['softmax_probabilities']
            print(epoch, softmax_probabilities.shape)
    else:
        cnn.load_model(model)

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
