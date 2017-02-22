#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys
import os

ProjectRootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ProjectRootPath)

import argparse
from itertools import izip_longest

import numpy as np
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
    for filename, save_filename in izip_longest(options.filenames, options.save_filenames, fillvalue=None):
        seen_number, accepted_number = get_drop_number(filename, options.dataset)

        if options.plot:
            delta_number = [seen - accepted for seen, accepted in zip(seen_number, accepted_number)]
            drop_number = [delta_number[0]] + [delta_number[i] - delta_number[i - 1]
                                               for i in range(1, len(delta_number))]

            # print(drop_number)

            plt.plot(drop_number, label=filename)
        else:
            if save_filename is None:
                save_filename = 'drop_num_{}'.format(filename)

            save_list(seen_number, os.path.join(DataPath, options.dataset, save_filename))

    if options.plot:
        plt.xlim(xmax=options.xmax)
        plt.ylim(ymax=options.ymax)

        plt.legend(loc='lower right')
        plt.show()


def get_drop_number_rank(filename, dataset='mnist', series_number=5):
    abs_filename = os.path.join(LogPath, dataset, filename)

    total_number = []

    rank_numbers = [[] for _ in range(series_number)]

    pick_index = None

    with open(abs_filename, 'r') as f:
        for line in f:
            if line.startswith('Part  (total'):
                words = line.split()

                if pick_index is None:
                    pick_index = np.linspace(3, len(words) - 1, series_number, dtype=int)

                total_number.append(int(words[2][:-2]))

                for i, rank_number in enumerate(rank_numbers):
                    rank_number.append(int(words[pick_index[i]]))

    return total_number, rank_numbers, pick_index


def plot_drop_number_rank(filename, **kwargs):
    dataset = kwargs.pop('dataset', 'mnist')
    plot_total = kwargs.pop('plot_total', False)
    series_number = kwargs.pop('series_number', 5)
    title = kwargs.pop('title', filename)

    vp2epoch = kwargs.pop('vp2epoch', {
        'mnist': 20 * 125.0 / 50000,
        'cifar10': 390 * 128.0 / 100000,
        'imdb': 16 * 200.0 / 22137,
    }[dataset])

    ymax = kwargs.pop('ymax', None)
    ymin = kwargs.pop('ymin', None)
    xmax = kwargs.pop('xmax', None)
    xmin = kwargs.pop('xmin', None)

    total, rank_numbers, pick_index = get_drop_number_rank(filename, dataset, series_number)

    xs = np.arange(len(total), dtype=float) * vp2epoch

    if plot_total:
        plt.plot(xs, total, label='$total$', linewidth=2.0)
    for rank_number, idx in zip(rank_numbers, pick_index):
        plt.plot(xs, rank_number, label='$Rank {}$'.format(idx - 3), linewidth=2.0)

    plt.xlim(xmin=xmin, xmax=xmax)
    plt.ylim(ymin=ymin, ymax=ymax)

    plt.legend(loc='upper left')
    plt.title('${}$'.format(title))

    plt.xlabel('$Epoch$', fontsize=18)
    plt.ylabel(r'$Filter\ Number$', fontsize=18)

    plt.show()


def main(args=None):
    parser = argparse.ArgumentParser(description='The drop number extractor')

    parser.add_argument('filenames', nargs='+', help='The log filenames')
    parser.add_argument('-d', '--dataset', action='store', dest='dataset', default='mnist',
                        help='The dataset (default is "mnist")')
    parser.add_argument('-o', action='append', nargs='+', dest='save_filenames', default=[],
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
    # main([
    #     '-p',
    #     'log-mnist-stochastic-lr-speed-NonC3Best.txt',
    #     'log-mnist-stochastic-lr-speed-NonC7Best.txt',
    #     'log-mnist-stochastic-lr-speed-NonC8Best.txt',
    #     'log-mnist-stochastic-lr-speed-NonC10Best.txt',
    # ])

    # plot_drop_number_rank(
    #     'log-cifar10-stochastic-lr-speed-NonC3Best_1.txt',
    #     dataset='cifar10',
    #     series_number=5,
    #     title='CIFAR-10\ NDF-REINFORCE\ LR',
    #     ymax=None,
    #     xmax=24,
    # )

    # plot_drop_number_rank(
    #     'log-imdb-stochastic-mlp-speed-NonC1Best.txt',
    #     dataset='imdb',
    #     series_number=5,
    #     title='IMDB\ NDF-REINFORCE\ MLP',
    #     xmax=10.5,
    # )

    plot_drop_number_rank(
        'log-mnist-stochastic-lr-speed-NonC8Best_1.txt',
        dataset='mnist',
        series_number=5,
        title='MNIST\ NDF-REINFORCE\ LR',
        xmax=65,
        plot_total=True,
    )

    pass
