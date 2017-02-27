# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

from libs.train import MNIST, CIFAR10, IMDB
from libs.utility.utils import process_before_train2

__author__ = 'fyabc'


def main():
    # Just ref them, or they may be optimized out by PyCharm.
    _ = CIFAR10, IMDB, MNIST

    # Set the configs (include dataset specific config), and return the dataset attributes.
    dataset_attr = process_before_train2()

    # Call the dataset main entry.
    eval('{}()'.format(dataset_attr.main_entry))


if __name__ == '__main__':
    main()
