# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys
import os
import argparse

ProjectRootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ProjectRootPath)

from itertools import count

import numpy as np
from scipy.interpolate import spline
import matplotlib.pyplot as plt

from config import LogPath
from utils import Curves, legend, average_list

__author__ = 'fyabc'


Interval = 4


def get_test_acc_list(filename, dataset='mnist', interval=Interval):
    abs_filename = os.path.join(LogPath, dataset, filename)

    filenames = [abs_filename]

    root, ext = os.path.splitext(abs_filename)

    for i in count(1):
        new_filename = '{}_{}{}'.format(root, i, ext)

        if os.path.exists(new_filename):
            filenames.append(new_filename)
        else:
            break

    results = []

    for filename in filenames:
        with open(filename, 'r') as f:
            results.append([
                float(line.split()[-1])
                for line in f
                if line.startswith('Test accuracy:')
            ])

    result = average_list(*results)

    result = [e for i, e in enumerate(result) if i % interval == 0]

    return result


def get_test_acc_lists(*filenames, **kwargs):
    dataset = kwargs.pop('dataset', 'mnist')
    interval = kwargs.pop('interval', Interval)

    return [get_test_acc_list(filename, dataset, interval) for filename in filenames]


def plot_mnist():
    raw = get_test_acc_list('log-mnist-raw-NonC1.txt')
    spl = get_test_acc_list('log-mnist-spl-NonC1.txt')

    speed = get_test_acc_list('log-mnist-stochastic-lr-speed-NonC2.txt')
    speed_best = get_test_acc_list('log-mnist-stochastic-lr-speed-NonC2Best.txt')
    delta_acc = get_test_acc_list('log-mnist-stochastic-lr-delta_acc-NonC2.txt')
    delta_acc_best = get_test_acc_list('log-mnist-stochastic-lr-delta_acc-NonC2Best.txt')

    plt.plot(raw, label='raw')
    plt.plot(spl, label='spl')

    plt.plot(speed, label='speed last (325)')
    plt.plot(speed_best, label='speed best (65)')
    plt.plot(delta_acc, label='delta acc last (503)')
    plt.plot(delta_acc_best, label='delta acc best (371)')

    plt.ylim(ymin=0.88, ymax=0.98)
    plt.xlim(xmax=1200 // Interval)

    plt.legend(loc='lower right')

    plt.show()


def plot_c_mnist():
    raw, best80, spl = get_test_acc_lists(
        'log-mnist-raw-Flip1.txt',
        'log-mnist-raw-FlipBest80.txt',
        'log-mnist-spl-ForceFlip1.txt',
    )

    # m0r0b2 = get_test_acc_list('log-mnist-stochastic-lr-m0r0b2.txt')
    # m2r_2b2 = get_test_acc_list('log-mnist-stochastic-lr-ForceFlip1.txt')
    # m_2r2b2 = get_test_acc_list('log-mnist-stochastic-lr-m_2r2b2.txt')
    # m_3r3b2 = get_test_acc_list('log-mnist-stochastic-lr-m_3r3b2.txt')
    # m_4r4b2 = get_test_acc_list('log-mnist-stochastic-lr-m_4r4b2.txt')

    # speed = get_test_acc_list('log-mnist-stochastic-lr-Flip2.txt')
    # speed_best = get_test_acc_list('log-mnist-stochastic-lr-speed-Flip2Best.txt')
    # delta_acc_best = get_test_acc_list('log-mnist-stochastic-lr-delta_acc-Flip2.txt')

    plt.plot(raw, label='raw')
    plt.plot(best80, label='best 80%')
    plt.plot(spl, label='spl')

    # plt.plot(speed, label='speed last (340)')
    # plt.plot(speed_best, label='speed best (286)')
    # plt.plot(delta_acc_best, label='delta acc best (213)')

    # plt.plot(m0r0b2, '--', label='m0r0b2')
    # plt.plot(m2r_2b2, '--', label='m2r-2b2')
    # plt.plot(m_2r2b2, '--', label='m-2r2b2')
    # plt.plot(m_3r3b2, '--', label='m-3r3b2')
    # plt.plot(m_4r4b2, '--', label='m-4r4b2')

    plt.ylim(ymin=0.87, ymax=0.96)
    plt.xlim(xmax=1200 // Interval)

    plt.legend(loc='lower right')

    plt.show()


def plot_accuracy_curve(title, style, y, vp_size, smooth, interval, maxlen):
    x = range(vp_size, 1 + len(y) * vp_size, vp_size)

    if interval > 1:
        x = [e for i, e in enumerate(x) if (i + 1) % interval == 0]
        y = [e for i, e in enumerate(y) if (i + 1) % interval == 0]

    min_len = min(len(x), len(y), maxlen)
    x, y = x[:min_len], y[:min_len]

    if smooth > 0:
        x_new = np.linspace(min(x), max(x), smooth)
        power_smooth = spline(x, y, x_new)
    else:
        x_new = x
        power_smooth = y

    plt.plot(x_new, power_smooth, linewidth=2.0, label=title, linestyle=style)


def plot_for_paper_mnist():
    plot_for_paper_all_mnist(
        'log-mnist-raw-NonC1.txt',
        'log-mnist-spl-NonC1.txt',
        'log-mnist-random_drop-speed-NonC3.txt',
        'log-mnist-stochastic-lr-speed-NonC3Best.txt',

        xmin=0,
        ymin=0.90,
        ymax=0.98,
    )


def plot_for_paper_c_mnist():
    plot_for_paper_all_mnist(
        'log-mnist-raw-Flip1.txt',
        'log-mnist-spl-ForceFlip1.txt',
        'log-mnist-random_drop-speed-Flip2.txt',
        'log-mnist-stochastic-lr-speed-Flip2Best.txt',
    )


def plot_for_paper_all_mnist(*filenames, **kwargs):
    interval = kwargs.pop('interval', 5)
    vp_size = kwargs.pop('vp_size', 2500)
    maxlen = kwargs.pop('maxlen', 400)
    smooth = kwargs.pop('smooth', 200)

    xmin = kwargs.pop('xmin', 160)
    xmax = kwargs.pop('xmax', 1200)
    ymin = kwargs.pop('ymin', 0.89)
    ymax = kwargs.pop('ymax', 0.96)

    raw, spl, random_drop, reinforce = get_test_acc_lists(*filenames, interval=1)

    plot_accuracy_curve(Curves[0].title, '-.', random_drop, vp_size, smooth, interval, maxlen)
    plot_accuracy_curve(Curves[2].title, '--', spl, vp_size, smooth, interval, maxlen)
    plot_accuracy_curve(Curves[3].title, '--', raw, vp_size, smooth, interval, maxlen)
    plot_accuracy_curve(Curves[5].title, '-', reinforce, vp_size, smooth, interval, maxlen)

    legend(use_ac=False)

    plt.xlim(xmin=xmin * vp_size, xmax=xmax * vp_size)
    plt.ylim(ymin=ymin, ymax=ymax)

    plt.show()


def plot_by_args(options):
    for filename in options.filename:
        data = get_test_acc_list(filename, options.dataset, options.interval)
        plt.plot(data, label=os.path.splitext(filename)[0])

    plt.xlim(xmax=options.xmax / options.interval)
    plt.ylim(ymin=options.ymin, ymax=options.ymax)

    plt.legend(loc='lower right')

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='The test accuracy plotter')

    parser.add_argument('filename', nargs='+')
    parser.add_argument('-d', '--dataset', action='store', dest='dataset', default='mnist',
                        help='The dataset (default is "mnist")')
    parser.add_argument('-i', '--interval', action='store', dest='interval', type=int, default=Interval,
                        help='The interval of validation point (default is $(Interval))')
    parser.add_argument('-y', '--ymin', action='store', dest='ymin', type=float, default=0.88,
                        help='The y min value (default is 0.88)')
    parser.add_argument('-Y', '--ymax', action='store', dest='ymax', type=float, default=0.98,
                        help='The y max value (default is 0.98)')
    parser.add_argument('-X', '--xmax', action='store', dest='xmax', type=int, default=1200,
                        help='The x max value before divided by interval (default is 1200)')

    options = parser.parse_args()

    plot_by_args(options)


if __name__ == '__main__':
    # plot_c_mnist()
    # plot_mnist()

    # main()

    # plot_for_paper_c_mnist()
    plot_for_paper_mnist()
    pass
