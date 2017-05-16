# -*- coding: utf-8 -*-

"""The main entry of training."""

from __future__ import print_function

from libs.train import *
from libs.utility.utils import process_before_train


def main():
    # Just ref them, or they may be optimized out by PyCharm.
    _ = CIFAR10

    # Set the configs (include dataset specific config), and return the dataset attributes.
    dataset_attr = process_before_train()

    # Call the dataset main entry.
    eval('{}()'.format(dataset_attr.main_entry))


if __name__ == '__main__':
    main()
