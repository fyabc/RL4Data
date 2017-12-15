#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import os
import sys
from collections import namedtuple

ProjectRootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ProjectRootPath)

import argparse

import numpy as np
import matplotlib.pyplot as plt

from libs.utility.config import LogPath


Data = namedtuple('Data', ['name', 'prefix_list', 'location'])


all_data = {
    'tr': Data('terminal reward', ('Real cost', 'TR'), -1),
    'b': Data('bias', '$    b', -1),
}

for i in range(15):
    all_data['w{}'.format(i)] = Data('W[{}]'.format(i), '$    W', i + 3)


def get_reward_list(filename, dataset='mnist', abspath=False, data_name='tr'):
    if not abspath:
        abs_filename = os.path.join(LogPath, dataset, filename)
    else:
        abs_filename = filename

    data = all_data[data_name]

    with open(abs_filename, 'r') as f:
        result = [
            float(line.split()[data.location])
            for line in f
            if line.strip().startswith(data.prefix_list)
        ]

    return result


def plot_by_args(options):
    colors = ['b', 'r', 'g', 'k', 'y']
    lines = ['-', '--', '-.', '.']

    for i, data_name in enumerate(options.data):
        reward_list = get_reward_list(options.filename, options.dataset, options.abspath, data_name)
        data = all_data[data_name]

        if options.ignore_zero:
            reward_list = [e if abs(e) > 1e-6 else None for e in reward_list]

        # Print the max and argmax.
        arg_max = np.argmax(reward_list)

        print('{} Total: {} episodes; Max: {} at episode {}'.format(
            data.name, len(reward_list), reward_list[arg_max], arg_max))

        if options.normalize:
            reward_list = np.array(reward_list)
            reward_list -= reward_list.mean()
            reward_list /= reward_list.std()

        style = '{}{}'.format(colors[i % len(colors)], lines[i // len(colors) % len(lines)])

        plt.plot(
            reward_list, style, label=data.name,
            linewidth=2 if data_name == 'tr' else 1,
        )

    plt.ylim(ymin=options.ymin, ymax=options.ymax)
    plt.grid()
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='The reward plotter')

    parser.add_argument('filename', help='The log filename')
    parser.add_argument('-d', '--dataset', action='store', dest='dataset', default='mnist',
                        help='The dataset (default is "%(default)s")')
    parser.add_argument('-i', '--ignore_zero', action='store_true', dest='ignore_zero', default=False,
                        help='Ignore the zero reward (default is %(default)s)')
    parser.add_argument('-y', '--ymin', action='store', dest='ymin', type=float, default=None,
                        help='The y min value (default is %(default)s)')
    parser.add_argument('-Y', '--ymax', action='store', dest='ymax', type=float, default=None,
                        help='The y max value (default is %(default)s)')
    parser.add_argument('-a', '--abspath', action='store_true', dest='abspath', default=False,
                        help='Filename is absolute path, (default is %(default)s)')
    parser.add_argument('-D', '--data', action='store', nargs='+', dest='data', default=['tr'],
                        help='The data to plot (default is "%(default)s")')
    parser.add_argument('-n', '--normalize', action='store_true', dest='normalize', default=False,
                        help='Add normalize on data, (default is %(default)s)')

    options = parser.parse_args()

    plot_by_args(options)


if __name__ == '__main__':
    main()
