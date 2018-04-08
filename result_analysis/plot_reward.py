#! /usr/bin/python
# -*- encoding: utf-8 -*-

"""Plot the reward or other data.

Some plots: curves are in "/result_analysis/curve"

    # mini-mnist-reward-smoothed.png
    $ python2 result_analysis\plot_reward.py -d mnist log-mnist-reinforce-fixed-acc-new.txt -D tr -X 102
    $ python2 result_analysis\plot_reward.py -d mnist log-mnist-reinforce-fixed-acc-new_full.txt -D ir -x 0 -X 50 -y 0.953  # noqa
    $ python2 result_analysis\plot_reward.py -a log\mnist\log-mnist-reinforce-fixed-speed-new_full.txt -D tr --running_avg 0.05
    $ python2 result_analysis\plot_reward.py -a log\mnist\log-mnist-reinforce-fixed-speed-new_full.txt -D delta_l2
"""

from __future__ import print_function

import os
import sys
from collections import namedtuple

ProjectRootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ProjectRootPath)

import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from libs.utility.config import LogPath
from libs.utility.plot_utils import move_avg, running_avg


Data = namedtuple('Data', ['name', 'prefix_list', 'location', 'y_label'])


def __l2(w, b):
    if isinstance(w, list):
        w = np.array(w)
    return np.sum(w ** 2) + b ** 2


def get_l2_norm(f):
    ws, bs = [], []
    for line in f:
        if line.startswith('$    W'):
            ws.append(list(map(float, line.strip().split()[3:])))
        elif line.startswith('$    b'):
            bs.append(float(line.strip().split()[-1]))

    return [__l2(w, b) for w, b in zip(ws, bs)]


def get_delta_l2_norm(f):
    ws, bs = [], []
    for line in f:
        if line.startswith('$    W'):
            ws.append(list(map(float, line.strip().split()[3:])))
        elif line.startswith('$    b'):
            bs.append(float(line.strip().split()[-1]))

    result = []
    current_w, current_b = np.zeros_like(ws[0]), 0.0
    for i in xrange(len(ws)):
        w = np.array(ws[i])
        b = bs[i]
        result.append(__l2(w - current_w, b - current_b))
        current_w, current_b = w, b

    # Pop first value
    result.pop(0)

    return result


all_data = {
    'tr': Data(r'$Terminal\ reward\ for\ training\ teacher\ model\ on\ MNIST$', ('Real cost', 'TR'), -1, r'$Terminal\ Reward$'),    # noqa
    # 'ir': Data('immediate reward', 'Average immediate', -1, r'$Terminal\ Reward$'),
    'ir': Data(r'$Terminal\ reward\ for\ training\ teacher\ model\ on\ MNIST$', 'Average immediate', -1, r'$Terminal\ Reward$'),    # noqa
    'b': Data('bias', '$    b', -1, r'$Bias\ value$'),
    'va': Data('valid accuracy', '$  best valid', -1, r'$Accuracy$'),
    'tea': Data('test accuracy', '$  best test', -2, r'$Accuracy$'),
    'l2': Data(r'$norm(\theta)$', get_l2_norm, None, r'$norm(\theta)$'),
    'delta_l2': Data(r'$norm(\Delta_\theta)\ for\ training\ teacher\ model\ on\ MNIST$', get_delta_l2_norm, None, r'$norm(\Delta_\theta)$')  # noqa
}

for i in range(15):
    all_data['w{}'.format(i)] = Data('W[{}]'.format(i), '$    W', i + 3, r'$Weight\ value$')


def get_reward_list(filename, dataset='mnist', abspath=False, data_name='tr'):
    if not abspath:
        abs_filename = os.path.join(LogPath, dataset, filename)
    else:
        abs_filename = filename

    data = all_data[data_name]

    if callable(data.prefix_list):
        with open(abs_filename, 'r') as f:
            return data.prefix_list(f)

    def _get_float(_line):
        val_str = _line.split()[data.location]
        if val_str == 'None':
            return 0.0
        return float(val_str)

    with open(abs_filename, 'r') as f:
        result = [
            _get_float(line)
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
        arg_max = np.argmax(reward_list[options.max_start:]) + options.max_start

        print('{} Total: {} episodes; Max: {} at episode {}'.format(
            data.name, len(reward_list), reward_list[arg_max], arg_max))

        if options.move_avg > 0:
            reward_list = move_avg(reward_list, options.move_avg, fill_start=0.0)
        if options.running_avg > 0.0:
            reward_list = running_avg(reward_list, move_ratio=options.running_avg, start_value=0.0)

        if len(options.data) > 1 and options.normalize:
            reward_list = np.array(reward_list)
            reward_list -= reward_list.mean()
            reward_list /= reward_list.std()

        style = '{}{}'.format(colors[i % len(colors)], lines[i // len(colors) % len(lines)])

        plt.plot(
            reward_list, style, label=data.name,
            linewidth=7.5,
        )

    # Set style.
    plt.xlabel(r'$Number\ of\ Episode$', fontsize=30)
    plt.ylabel(data.y_label, fontsize=30)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=24)

    plt.xlim(xmin=options.xmin, xmax=options.xmax)
    plt.ylim(ymin=options.ymin, ymax=options.ymax)
    plt.grid(True, axis='both', linestyle='--')
    plt.legend(loc='best', fontsize=28, borderpad=0.2, labelspacing=0.2, handletextpad=0.2, borderaxespad=0.2)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='The reward plotter')

    parser.add_argument('filename', help='The log filename')
    parser.add_argument('-d', '--dataset', action='store', dest='dataset', default='mnist',
                        help='The dataset (default is "%(default)s")')
    parser.add_argument('-i', '--ignore_zero', action='store_true', dest='ignore_zero', default=False,
                        help='Ignore the zero reward (default is %(default)s)')
    parser.add_argument('-x', '--xmin', action='store', dest='xmin', type=float, default=None,
                        help='The x min value (default is %(default)s)')
    parser.add_argument('-X', '--xmax', action='store', dest='xmax', type=float, default=None,
                        help='The x max value (default is %(default)s)')
    parser.add_argument('-y', '--ymin', action='store', dest='ymin', type=float, default=None,
                        help='The y min value (default is %(default)s)')
    parser.add_argument('-Y', '--ymax', action='store', dest='ymax', type=float, default=None,
                        help='The y max value (default is %(default)s)')
    parser.add_argument('-a', '--abspath', action='store_true', dest='abspath', default=False,
                        help='Filename is absolute path, (default is %(default)s)')
    parser.add_argument('-D', '--data', action='store', nargs='+', dest='data', default=['tr'],
                        choices=all_data.keys(), help='The data to plot (default is "%(default)s")')
    parser.add_argument('-n', '--normalize', action='store_true', dest='normalize', default=False,
                        help='Add normalize on data, (default is %(default)s)')
    parser.add_argument('--move_avg', action='store', dest='move_avg', default=0, type=int,
                        help='Moving average, default is %(default)s')
    parser.add_argument('--running_avg', action='store', dest='running_avg', default=0.0, type=float,
                        help='Running average, default is %(default)s')
    parser.add_argument('-L', '--no-latex', action='store_false', dest='latex', default=True,
                        help='Turn off LaTeX rendering, which will spend several seconds')
    parser.add_argument('--max-start', action='store', dest='max_start', default=0, type=int,
                        help='The start index to calculate the max value, default is %(default)s')

    options = parser.parse_args()

    if options.latex:
        try:
            # [NOTE]: Must add this after matplotlib 2.0.0.
            matplotlib.rcParams['text.usetex'] = True
        except KeyError:
            pass

    plot_by_args(options)


if __name__ == '__main__':
    main()
