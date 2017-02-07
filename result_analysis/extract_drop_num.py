#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys
import os

ProjectRootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ProjectRootPath)

import argparse

from config import LogPath, DataPath
from utils import save_list

__author__ = 'fyabc'


def get_drop_number(filename, dataset='mnist'):
    abs_filename = os.path.join(LogPath, dataset, filename)

    with open(abs_filename, 'r') as f:
        result = [
            int(line.split()[-2])
            for line in f
            if line.startswith('Number of accepted cases')
        ]

    return result


def plot_by_args(options):
    drop_number = get_drop_number(options.filename, options.dataset)

    save_list(drop_number, os.path.join(DataPath, options.dataset, 'drop_num_{}'.format(options.filename)))


def main():
    parser = argparse.ArgumentParser(description='The drop number extractor')

    parser.add_argument('filename', help='The log filename')
    parser.add_argument('-d', '--dataset', action='store', dest='dataset', default='mnist',
                        help='The dataset (default is "mnist")')

    options = parser.parse_args()

    plot_by_args(options)


if __name__ == '__main__':
    main()
