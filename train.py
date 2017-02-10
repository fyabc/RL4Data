# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys

import train_CIFAR10
import train_IMDB
import train_MNIST

from utils import process_before_train2

__author__ = 'fyabc'


def main():
    # Set the configs (include dataset specific config), and return the dataset attributes.
    dataset_attr = process_before_train2()

    # Call the dataset main entry.
    eval('{}()'.format(dataset_attr.main_entry))


if __name__ == '__main__':
    main()
