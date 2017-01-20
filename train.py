# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys

import train_CIFAR10
import train_IMDB
import train_MNIST

from utils import process_before_train2

__author__ = 'fyabc'


def main2():
    # Set the configs (include dataset specific config), and return the dataset attributes.
    dataset_attr = process_before_train2()

    # Call the dataset main entry.
    eval('{}()'.format(dataset_attr.main_entry))


def main():
    Datasets = [
        'CIFAR10',
        'IMDB',
        'MNIST',
    ]

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


if __name__ == '__main__':
    main()
