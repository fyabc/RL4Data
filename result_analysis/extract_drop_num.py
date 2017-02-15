#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys
import os

ProjectRootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ProjectRootPath)

import argparse

import matplotlib.pyplot as plt

from config import LogPath, DataPath
from utils import save_list

__author__ = 'fyabc'


def get_drop_number(filename, dataset='mnist'):
    abs_filename = os.path.join(LogPath, dataset, filename)

    result = []
    result_accepted = []

    with open(abs_filename, 'r') as f:
        for line in f:
            if line.startswith(('Number of accepted cases', 'NAC:')):
                result.append(int(line.split()[-2]))
                result_accepted.append(int(line.split()[-4]))

    return result, result_accepted


def plot_by_args(options):
    seen_number, accepted_number = get_drop_number(options.filename, options.dataset)

    if options.plot:
        delta_number = [seen - accepted for seen, accepted in zip(seen_number, accepted_number)]
        drop_number = [delta_number[0]] + [delta_number[i] - delta_number[i - 1] for i in range(1, len(delta_number))]

        print(drop_number)

        plt.plot(drop_number, label=options.filename)

        plt.xlim(xmax=options.xmax)
        plt.ylim(ymax=options.ymax)

        plt.legend(loc='lower right')

        plt.show()
    else:
        if options.save_filename is None:
            options.save_filename = 'drop_num_{}'.format(options.filename)

        save_list(seen_number, os.path.join(DataPath, options.dataset, options.save_filename))


def main(args=None):
    parser = argparse.ArgumentParser(description='The drop number extractor')

    parser.add_argument('filename', help='The log filename')
    parser.add_argument('-d', '--dataset', action='store', dest='dataset', default='mnist',
                        help='The dataset (default is "mnist")')
    parser.add_argument('-o', action='store', dest='save_filename', default=None,
                        help='The save filename (default is "drop_num_$(filename)")')
    parser.add_argument('-p', '--plot', action='store_true', dest='plot', default=False,
                        help='Plot the drop number instead of dump it (default is False)')
    parser.add_argument('-X', '--xmax', action='store', dest='xmax', type=int, default=None,
                        help='The x max value before divided by interval (default is None)')
    parser.add_argument('-Y', '--ymax', action='store', dest='ymax', type=float, default=None,
                        help='The y max value (default is None)')

    options = parser.parse_args(args)

    plot_by_args(options)


if __name__ == '__main__':
    main(['-p', 'log-cifar10-stochastic-lr-speed-Flip3Best_1.txt', '-d', 'cifar10'])
