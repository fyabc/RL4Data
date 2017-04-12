#! /usr/bin/python
# -*- encoding: utf-8 -*-

"""The new version of training file."""

from __future__ import print_function, unicode_literals

import argparse

from libs.train import *
from libs.utility.utils import preprocess_v2

__author__ = 'fyabc'


def main():
    parser = argparse.ArgumentParser(description='The version 2 of training entry.')

    parser.add_argument('dataset', dest='dataset',
                        help='The dataset, ignore case.')
    parser.add_argument('job_name', dest='job_name',
                        help='The job name.')
    parser.add_argument('-c', '--config', action='store', metavar='filename', dest='config',
                        help='Config filename. If it is a new job, must given. '
                             'If it is an exist job, it will overwrite the old config file.')

    args = parser.parse_args()

    # Just ref them, or they may be optimized out by PyCharm.
    _ = CIFAR10

    # Set the configs (include dataset specific config), and return the dataset attributes.
    config, dataset_attr = preprocess_v2(args)

    # Call the dataset main entry.
    eval('{}()'.format(dataset_attr.main_entry))


if __name__ == '__main__':
    main()
