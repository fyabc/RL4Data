#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import os
import sys

ProjectRootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ProjectRootPath)

import argparse
from itertools import izip_longest

import numpy as np
import matplotlib.pyplot as plt

from libs.utility.config import LogPath, DataPath
from utils import save_list, move_avg, CFG, pick_interval


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

    rank_size = None
    pick_start = None
    pick_end = None

    with open(abs_filename, 'r') as f:
        for line in f:
            if line.startswith('Part  (total'):
                words = line.split()

                if rank_size is None:
                    rank_size = len(words) - 3
                    pick_start = np.linspace(0, rank_size, series_number, dtype=int, endpoint=False)
                    pick_end = [pick_start[i + 1] for i in range(series_number - 1)] + [rank_size]

                total_number.append(int(words[2][:-2]))

                for i, rank_number in enumerate(rank_numbers):
                    rank_number.append(sum(int(word) for word in words[3 + pick_start[i]:3 + pick_end[i]]))

    return total_number, rank_numbers, pick_start, pick_end, rank_size


def plot_drop_number_rank(filename, kwargs):
    dataset = kwargs.pop('dataset', 'mnist')
    plot_total = kwargs.pop('plot_total', False)
    series_number = kwargs.pop('series_number', 5)
    title = kwargs.pop('title', filename)
    mv_avg = kwargs.pop('mv_avg', False)
    interval = kwargs.pop('interval', 1)
    legend_loc = kwargs.pop('legend_loc', 'upper left')
    use_legend = kwargs.pop('use_legend', False)
    style = kwargs.pop('style', None)

    vp2epoch = kwargs.pop('vp2epoch', {
        'mnist': 20 * 125.0 / 50000,
        'cifar10': 390 * 128.0 / 100000,
        'imdb': 16 * 200.0 / 22137,
    }[dataset])

    ymax = kwargs.pop('ymax', None)
    ymin = kwargs.pop('ymin', None)
    xmax = kwargs.pop('xmax', None)
    xmin = kwargs.pop('xmin', None)

    line_width = kwargs.pop('line_width', CFG['linewidth'])

    line_styles = ['-', '--', '-.', 'o-', '*-']

    total, rank_numbers, pick_start, pick_end, rank_size = get_drop_number_rank(filename, dataset, series_number)

    xs = np.arange(len(total), dtype=float) * vp2epoch
    xs = pick_interval(xs, interval)

    if plot_total:
        if mv_avg is not False:
            total = move_avg(total, mv_avg)

        total = pick_interval(total, interval)

        plt.plot(xs, total, label='$Total$',
                 linewidth=line_width, markersize=CFG['markersize'])

    colors = ['blue', 'green', 'red', 'cyan', 'magenta']

    for i, rank_number in reversed(list(enumerate(rank_numbers))):
        if mv_avg is not False:
            rank_number = move_avg(rank_number, mv_avg)

        rank_number = pick_interval(rank_number, interval)

        if i == 0:
            more_str = r'\ (Easiest)'
        elif i == series_number - 1:
            more_str = r'\ (Hardest)'
        else:
            more_str = r''
        label = r'$Bucket\ {}{}$'.format(series_number - i, more_str)
        # label = r'$Bucket\ {} \sim {}$'.format(rank_size + 1 - pick_end[i], rank_size - pick_start[i])
        # label = r'$Hardness\ {}$'.format(i + 1)

        plt.plot(xs, rank_number, line_styles[i],
                 label=label,
                 linewidth=line_width, markersize=CFG['markersize'], color=colors[i])

    plt.xlim(xmin=xmin, xmax=xmax)
    plt.ylim(ymin=ymin, ymax=ymax)

    if use_legend:
        plt.legend(loc=legend_loc, fontsize=28,
                   borderpad=0.2, labelspacing=0.2, handletextpad=0.2, borderaxespad=0.2)
    # plt.title('${}$'.format(title), fontsize=40)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=24)

    plt.xlabel('$Epoch$', fontsize=30)
    plt.ylabel(r'$Filter\ Number$', fontsize=30)

    plt.grid(True, axis='both', linestyle='--')

    plt.show()


def argparse_main(kwargs, args=None):
    parser = argparse.ArgumentParser(description='The drop number extractor')

    parser.add_argument('filenames', nargs='+', help='The log filenames')
    parser.add_argument('-d', '--dataset', action='store', dest='dataset', default='mnist',
                        help='The dataset (default is "mnist")')
    parser.add_argument('-o', nargs='+', dest='save_filenames', default=[],
                        help='The save filename (default is "drop_num_$(filename)")')
    parser.add_argument('-p', '--plot', action='store_true', dest='plot', default=False,
                        help='Plot the drop number instead of dump it (default is False)')
    parser.add_argument('-X', '--xmax', action='store', dest='xmax', type=int, default=None,
                        help='The x max value before divided by interval (default is None)')
    parser.add_argument('-Y', '--ymax', action='store', dest='ymax', type=float, default=None,
                        help='The y max value (default is None)')

    options = parser.parse_args(args)

    plot_by_args(options)


def plot_cifar10(kwargs):
    kwargs.update(dict(
        dataset='cifar10',
        series_number=5,
        title='CIFAR-10\ NDF-REINFORCE\ LR',
        ymax=45000,
        xmax=24,
    ))

    plot_drop_number_rank(
        'log-cifar10-stochastic-lr-speed-NonC3Best_1.txt',
        kwargs,
    )


def plot_cifar10_half(kwargs):
    kwargs.update(dict(
        dataset='cifar10',
        series_number=5,
        title='CIFAR-10\ NDF-REINFORCE\ LR',
        ymax=28500,
        xmax=24,
    ))

    plot_drop_number_rank(
        'log-cifar10-stochastic-lr-speed-Cifar10NonC4Best_Half.txt',
        kwargs,
    )


def plot_imdb(kwargs):
    kwargs.update(dict(
        dataset='imdb',
        series_number=5,
        title='IMDB\ NDF-REINFORCE\ LR',
        # xmax=5.420788724759452,
        # xmax=12,
        xmax=16,
        mv_avg=2,
        legend_loc='upper right',
    ))

    plot_drop_number_rank(
        'log-imdb-stochastic-lr-speed-NonC_Old2_2.txt',
        kwargs,
    )


def plot_imdb_half(kwargs):
    kwargs.update(dict(
        dataset='imdb',
        series_number=5,
        title='IMDB\ NDF-REINFORCE\ LR',
        # xmax=5.420788724759452,
        # xmax=12,
        xmax=16,
        mv_avg=2,
        legend_loc='upper right',
    ))

    plot_drop_number_rank(
        'log-imdb-stochastic-lr-speed-Half_NonC_Old2.txt',
        kwargs,
    )


def plot_mnist(kwargs):
    kwargs.update(dict(
        dataset='mnist',
        series_number=5,
        title='MNIST\ NDF-REINFORCE\ LR',
        xmax=64,
        plot_total=False,

        interval=10,
    ))

    plot_drop_number_rank(
        'log-mnist-stochastic-lr-speed-NonC8Best_1.txt',
        kwargs,
    )


def plot_mnist_half(kwargs):
    kwargs.update(dict(
        dataset='mnist',
        series_number=5,
        title='MNIST\ NDF-REINFORCE\ LR',
        xmax=64,
        plot_total=False,

        interval=10,
    ))

    plot_drop_number_rank(
        'log-mnist-stochastic-lr-speed-Half_MnistNonC8Best.txt',
        kwargs,
    )


def main():
    # argparse_main([
    #     '-p',
    #     'log-mnist-stochastic-lr-speed-NonC3Best.txt',
    #     'log-mnist-stochastic-lr-speed-NonC7Best.txt',
    #     'log-mnist-stochastic-lr-speed-NonC8Best.txt',
    #     'log-mnist-stochastic-lr-speed-NonC10Best.txt',
    # ])

    kwargs = dict(
        use_legend=True,
    )

    {
        'imdb': plot_imdb,
        'imdb-half': plot_imdb_half,
        'cifar10': plot_cifar10,
        'cifar10-half': plot_cifar10_half,
        'mnist': plot_mnist,
        'mnist-half': plot_mnist_half,
        'argparse_main': argparse_main,
    }['imdb-half'](kwargs)

    pass


if __name__ == '__main__':
    main()
