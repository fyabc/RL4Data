#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys
import os
import argparse

ProjectRootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ProjectRootPath)

import matplotlib.pyplot as plt

from config import LogPath

__author__ = 'fyabc'


def get_reward_list(filename, dataset='mnist'):
    abs_filename = os.path.join(LogPath, dataset, filename)

    with open(abs_filename, 'r') as f:
        result = [
            float(line.split()[-1])
            for line in f
            if line.startswith('Real cost')
        ]

    return result


def plot_by_args(options):
    reward_list = get_reward_list(options.filename, options.dataset)

    if options.ignore_zero:
        reward_list = [e if abs(e) > 1e-6 else None for e in reward_list]

    plt.plot(reward_list, label='terminal reward')

    plt.ylim(ymin=options.ymin, ymax=options.ymax)

    plt.legend(loc='lower right')

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='The reward plotter')

    parser.add_argument('filename', help='The log filename')
    parser.add_argument('-d', '--dataset', action='store', dest='dataset', default='mnist',
                        help='The dataset (default is "mnist")')
    parser.add_argument('-I', '--no_ignore_zero', action='store_false', dest='ignore_zero', default=True,
                        help='Do not ignore the zero reward (default is False)')
    parser.add_argument('-y', '--ymin', action='store', dest='ymin', type=float, default=None,
                        help='The y min value (default is None)')
    parser.add_argument('-Y', '--ymax', action='store', dest='ymax', type=float, default=None,
                        help='The y max value (default is None)')

    options = parser.parse_args()

    plot_by_args(options)


if __name__ == '__main__':
    main()

