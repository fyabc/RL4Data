# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys

import train_CIFAR10
import train_IMDB
import train_MNIST

__author__ = 'fyabc'


Datasets = [
    'CIFAR10',
    'IMDB',
    'MNIST',
]


if __name__ == '__main__':
    dataset = 'CIFAR10'

    if len(sys.argv) >= 2:
        if sys.argv[1] in Datasets:
            dataset = sys.argv[1]

    print('Dataset: {}'.format(dataset))

    # Dict of all tasks.
    {
        'CIFAR10': train_CIFAR10.main,
        'IMDB': train_IMDB.main,
        'MNIST': train_MNIST.main,
    }[dataset](sys.argv)
