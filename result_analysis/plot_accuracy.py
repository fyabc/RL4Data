# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys
import os
import argparse

ProjectRootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ProjectRootPath)

import matplotlib.pyplot as plt

from config import LogPath

__author__ = 'fyabc'


Interval = 4


def get_test_acc_list(filename, dataset='mnist', interval=Interval):
    abs_filename = os.path.join(LogPath, dataset, filename)

    with open(abs_filename, 'r') as f:
        result = [
            float(line.split()[-1])
            for line in f
            if line.startswith('Test accuracy:')
        ]

    result = [e for i, e in enumerate(result) if i % interval == 0]

    return result


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
    raw = get_test_acc_list('log-mnist-raw-Flip1.txt')
    best80 = get_test_acc_list('log-mnist-raw-FlipBest80.txt')
    spl = get_test_acc_list('log-mnist-spl-ForceFlip1.txt')

    m0r0b2 = get_test_acc_list('log-mnist-stochastic-lr-m0r0b2.txt')
    m2r_2b2 = get_test_acc_list('log-mnist-stochastic-lr-ForceFlip1.txt')
    m_2r2b2 = get_test_acc_list('log-mnist-stochastic-lr-m_2r2b2.txt')
    m_3r3b2 = get_test_acc_list('log-mnist-stochastic-lr-m_3r3b2.txt')
    m_4r4b2 = get_test_acc_list('log-mnist-stochastic-lr-m_4r4b2.txt')

    speed = get_test_acc_list('log-mnist-stochastic-lr-Flip2.txt')
    speed_best = get_test_acc_list('log-mnist-stochastic-lr-speed-Flip2Best.txt')
    delta_acc_best = get_test_acc_list('log-mnist-stochastic-lr-delta_acc-Flip2.txt')

    plt.plot(raw, label='raw')
    # plt.plot(best80, label='best 80%')
    plt.plot(spl, label='spl')

    plt.plot(speed, label='speed last (340)')
    plt.plot(speed_best, label='speed best (286)')
    plt.plot(delta_acc_best, label='delta acc best (213)')

    # plt.plot(m0r0b2, '--', label='m0r0b2')
    # plt.plot(m2r_2b2, '--', label='m2r-2b2')
    # plt.plot(m_2r2b2, '--', label='m-2r2b2')
    # plt.plot(m_3r3b2, '--', label='m-3r3b2')
    # plt.plot(m_4r4b2, '--', label='m-4r4b2')

    plt.ylim(ymin=0.87, ymax=0.96)
    plt.xlim(xmax=1200 // Interval)

    plt.legend(loc='lower right')

    plt.show()


def plot_by_args(options):
    for filename in options.filenames:
        data = get_test_acc_list(filename, options.dataset, options.interval)
        plt.plot(data, label=os.path.splitext(filename)[0])

    plt.xlim(xmax=options.xmax / options.interval)
    plt.ylim(ymin=options.ymin, ymax=options.ymax)

    plt.legend(loc='lower right')

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='The test accuracy plotter')

    parser.add_argument('filename', nargs='+', dest='filenames')
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

    main()
    pass
