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
from utils import Curves, legend, average_list, avg

__author__ = 'fyabc'


Interval = 4


def get_data_list(filename, dataset='mnist', interval=Interval, start_tags=('Test accuracy:', 'TeA:')):
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
                if line.startswith(start_tags)
            ])

    result = average_list(*results)

    result = [e for i, e in enumerate(result) if i % interval == 0]

    return result


def get_test_acc_lists(*filenames, **kwargs):
    dataset = kwargs.pop('dataset', 'mnist')
    interval = kwargs.pop('interval', Interval)

    return [get_data_list(filename, dataset, interval) for filename in filenames]


def get_train_loss_lists(*filenames, **kwargs):
    dataset = kwargs.pop('dataset', 'mnist')
    interval = kwargs.pop('interval', Interval)

    return [get_data_list(filename, dataset, interval, 'tL') for filename in filenames]


def plot_mnist():
    raw = get_data_list('log-mnist-raw-NonC1.txt')
    spl = get_data_list('log-mnist-spl-NonC1.txt')

    speed = get_data_list('log-mnist-stochastic-lr-speed-NonC2.txt')
    speed_best = get_data_list('log-mnist-stochastic-lr-speed-NonC2Best.txt')
    delta_acc = get_data_list('log-mnist-stochastic-lr-delta_acc-NonC2.txt')
    delta_acc_best = get_data_list('log-mnist-stochastic-lr-delta_acc-NonC2Best.txt')

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


def plot_accuracy_curve(title, style, y, vp_size, smooth, interval, maxlen, **kwargs):
    mv_avg = kwargs.pop('mv_avg', False)

    x = range(vp_size, 1 + len(y) * vp_size, vp_size)

    if interval > 1:
        x = [e for i, e in enumerate(x) if (i + 1) % interval == 0]
        y = [e for i, e in enumerate(y) if (i + 1) % interval == 0]

    min_len = min(len(x), len(y), maxlen)
    x, y = x[:min_len], y[:min_len]

    if mv_avg is not False:
        y_new = [avg(y[max(i - mv_avg, 0):i + 1]) for i in range(len(y))]
        y = y_new

    if smooth > 0:
        try:
            x_new = np.linspace(min(x), max(x), smooth)
        except ValueError as e:
            print('Plot error:', title, e)
            return
        power_smooth = spline(x, y, x_new)
    else:
        x_new = x
        power_smooth = y

    plt.plot(x_new, power_smooth, style, label=title, **kwargs)


def plot_for_paper_all(*filenames, **kwargs):
    dataset = kwargs.pop('dataset', 'mnist')
    interval = kwargs.pop('interval', 5)
    vp_size = kwargs.pop('vp_size', 2500)
    maxlen = kwargs.pop('maxlen', 400)
    smooth = kwargs.pop('smooth', 200)
    spl_cfg = kwargs.pop('spl_cfg', [165])
    speed_cfg = kwargs.pop('speed_cfg', ['-'])
    line_width = kwargs.pop('line_width', 2.5)

    spl_count = len(spl_cfg)
    speed_count = len(speed_cfg)

    if False:
        figure, (axis1, axis2) = plt.subplots(1, 2)

        plt.sca(axis1)

    # Plot test accuracy
    plt.title(r'$Test\ Accuracy$', fontsize=18)

    xmin = kwargs.pop('xmin', 160)
    xmax = kwargs.pop('xmax', 1200)
    ymin = kwargs.pop('ymin', 0.89)
    ymax = kwargs.pop('ymax', 0.96)
    mv_avg = kwargs.pop('mv_avg', False)

    test_acc_lists = get_test_acc_lists(*filenames, interval=1, dataset=dataset)
    raw = test_acc_lists[0]
    spls = test_acc_lists[1:1 + spl_count]
    random_drop = test_acc_lists[1 + spl_count]
    reinforces = test_acc_lists[-speed_count:]

    # The color should be fixed.
    plot_accuracy_curve(Curves[0].title, 'b-', random_drop, vp_size, smooth, interval, maxlen,
                        linewidth=line_width, mv_avg=mv_avg)

    spl_lines = ['g--', 'm--', 'y--']

    for i, spl in enumerate(spls):
        plot_accuracy_curve(r'$SPL-{}$'.format(spl_cfg[i]),
                            spl_lines[i], spl, vp_size, smooth, interval, maxlen,
                            linewidth=line_width, mv_avg=mv_avg)

    plot_accuracy_curve(Curves[3].title, 'r-.', raw, vp_size, smooth, interval, maxlen,
                        linewidth=line_width + 1, mv_avg=mv_avg)

    reinforce_line_style = '-'
    reinforce_lines = ['c', 'm', 'y', 'g', 'k', 'w']

    for i, reinforce in enumerate(reinforces):
        plot_accuracy_curve(r'$NDF-REINFORCE-{}$'.format(speed_cfg[i]),
                            reinforce_lines[i] + reinforce_line_style, reinforce, vp_size, smooth, interval, maxlen,
                            linewidth=line_width, mv_avg=mv_avg)

    legend(use_ac=False, spl_count=spl_count, speed_count=speed_count, n_rows=3)

    plt.xlim(xmin=xmin * vp_size, xmax=xmax * vp_size)
    plt.ylim(ymin=ymin, ymax=ymax)
    plt.grid(True, axis='y', linestyle='--')

    # End plot test accuracy

    if False:
        # Plot training loss
        plt.sca(axis2)
        plt.title(r'$Training\ Loss$')

        interval = kwargs.pop('interval2', 80)
        vp_size = kwargs.pop('vp_size2', 1)
        smooth = kwargs.pop('smooth2', 0)
        maxlen = kwargs.pop('maxlen2', 2000)
        # line_width = 0.7

        xmin = kwargs.pop('xmin2', None)
        xmax = kwargs.pop('xmax2', None)
        ymin = kwargs.pop('ymin2', None)
        ymax = kwargs.pop('ymax2', 0.75)

        mv_avg = kwargs.pop('mv_avg2', 5)

        train_loss_lists = get_train_loss_lists(*filenames, interval=1, dataset=dataset)

        raw = train_loss_lists[0]
        spls = train_loss_lists[1:1 + spl_count]
        random_drop = train_loss_lists[1 + spl_count]
        reinforces = train_loss_lists[-speed_count:]

        plot_accuracy_curve(Curves[3].title, 'b-', random_drop, vp_size, smooth, interval, maxlen,
                            linewidth=line_width, mv_avg=mv_avg)

        for i, spl in enumerate(spls):
            plot_accuracy_curve(r'$SPL-{}$'.format(spl_cfg[i]),
                                spl_lines[i], spl, vp_size, smooth, interval, maxlen,
                                linewidth=line_width, mv_avg=mv_avg)

        plot_accuracy_curve(Curves[3].title, 'r-.', raw, vp_size, smooth, interval, maxlen,
                            linewidth=line_width + 1, mv_avg=mv_avg)

        for i, reinforce in enumerate(reinforces):
            plot_accuracy_curve(r'$NDF-REINFORCE-{}$'.format(speed_cfg[i]),
                                reinforce_lines[i] + reinforce_line_style, reinforce, vp_size, smooth, interval, maxlen,
                                linewidth=line_width, mv_avg=mv_avg)

        plt.xlim(xmin=xmin, xmax=xmax)
        plt.ylim(ymin=ymin, ymax=ymax)
        plt.grid(True, axis='y', linestyle='--')

        # End plot training loss

    # figure.tight_layout()
    # figure.set_size_inches(20, 5)

    plt.show()


def plot_for_paper_mnist():
    plot_for_paper_all(
        'log-mnist-raw-NonC1.txt',
        'log-mnist-spl-NonC5.txt',
        'log-mnist-spl-NonC4.txt',
        'log-mnist-spl-NonC6.txt',
        'log-mnist-random_drop-speed-NonC3.txt',
        'log-mnist-stochastic-lr-speed-NonC3Best.txt',
        'log-mnist-stochastic-lr-speed-NonC7Best.txt',
        'log-mnist-stochastic-lr-speed-NonC8Best.txt',
        'log-mnist-stochastic-lr-speed-NonC10Best.txt',
        # 'log-mnist-stochastic-lr-speed-NonC11Best.txt',

        xmin=130,
        xmax=1100,
        ymin=0.93,
        ymax=0.985,
        interval=2,
        maxlen=600,
        smooth=800,

        interval2=200,
        xmax2=200000,
        smooth2=200,
        mv_avg2=20,

        spl_cfg=[80, 120, 160],
        speed_cfg=['lrNonC3', 'lrNonC7', 'lrNonC8', 'lrNonC10'],
    )


def plot_for_paper_c_mnist():
    plot_for_paper_all(
        'log-mnist-raw-Flip1.txt',
        'log-mnist-spl-ForceFlip1.txt',
        'log-mnist-random_drop-speed-Flip2.txt',
        'log-mnist-stochastic-lr-speed-Flip2Best.txt',
    )


def plot_for_paper_cifar():
    plot_for_paper_all(
        'log-cifar10-raw-NonC1.txt',

        'log-cifar10-spl-NonC1.txt',
        'log-cifar10-spl-NonC60.txt',
        'log-cifar10-spl-NonC180.txt',

        'log-cifar10-random_drop-speed-NonC2.txt',

        'log-cifar10-stochastic-lr-speed-NonC2Best_1.txt',
        'log-cifar10-stochastic-lr-speed-NonC3Best.txt',
        'log-cifar10-stochastic-lr-speed-NonC4Best.txt',

        dataset='cifar10',
        xmin=0,
        xmax=99,
        ymin=0.6,
        ymax=0.95,
        interval=1,
        vp_size=390 * 128,
        smooth=800,

        ymin2=None,
        ymax2=None,
        xmax2=40000,

        spl_cfg=[120, 60, 180],
        speed_cfg=['lrNonC2', 'lrNonC3', 'lrNonC4']
    )


def plot_for_paper_c_cifar():
    plot_for_paper_all(
        'log-cifar10-raw-Flip1.txt',
        'log-cifar10-spl-Flip60.txt',
        'log-cifar10-spl-Flip1.txt',
        'log-cifar10-spl-Flip180.txt',
        # 'log-cifar10-raw-Flip1.txt',
        'log-cifar10-random_drop-speed-Flip3.txt',
        # 'log-cifar10-raw-Flip1.txt',
        # 'log-cifar10-stochastic-lr-delta_acc-Flip2Best_1.txt',
        'log-cifar10-stochastic-lr-speed-Flip3Best.txt',
        # 'log-cifar10-raw-Flip1.txt',

        dataset='cifar10',
        xmin=0,
        xmax=86,
        ymin=0.6,
        ymax=0.9,
        interval=1,
        vp_size=390 * 128,
        smooth=400,

        spl_cfg=[60, 124, 180],
        speed_cfg=['1'],
    )


def plot_for_paper_imdb():
    plot_for_paper_all(
        'log-imdb-raw-NonC1.txt',
        'log-imdb-spl-NonC1.txt',
        'log-imdb-spl-NonC2.txt',
        'log-imdb-spl-NonC3.txt',
        'log-imdb-random_drop-lr-speed-NonC_old2.txt',
        'log-imdb-stochastic-lr-speed-NonC_Old2.txt',
        'log-imdb-stochastic-mlp-speed-NonC1Best.txt',
        'log-imdb-stochastic-mlp-speed-NonC2Best.txt',

        dataset='imdb',
        xmin=0,
        xmax=37.5,
        ymin=0.45,
        ymax=0.92,
        interval=1,
        vp_size=200 * 16,
        # smooth=800,
        mv_avg=5,

        ymin2=None,
        ymax2=None,
        xmax2=40000,
        interval2=200,
        smooth2=800,
        mv_avg2=5,

        speed_cfg=['lrNonCOld', 'mlpNonC1', 'mlpNonC2'],
        spl_cfg=[40, 80, 160],
    )


def plot_by_args(options):
    for filename in options.filename:
        data = get_data_list(filename, options.dataset, options.interval)
        plt.plot(data, label=os.path.splitext(filename)[0])

    plt.xlim(xmax=options.xmax / options.interval)
    plt.ylim(ymin=options.ymin, ymax=options.ymax)

    plt.legend(loc='lower right')

    plt.grid(True, axis='y', linestyle='--')

    plt.show()


def main(args=None):
    parser = argparse.ArgumentParser(description='The test accuracy plotter')

    parser.add_argument('filename', nargs='*')
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
    parser.add_argument('-b', '--builtin', action='store', dest='builtin', default=None,
                        help='Plot the builtin curve (default is None, candidates are (c)cifar10/mnist)')

    options = parser.parse_args(args)

    if options.builtin is None:
        plot_by_args(options)
    else:
        {
            'mnist': plot_for_paper_mnist,
            'c-mnist': plot_for_paper_c_mnist,
            'cifar10': plot_for_paper_cifar,
            'c-cifar10': plot_for_paper_c_cifar,
            'imdb': plot_for_paper_imdb,
        }[options.builtin]()


if __name__ == '__main__':
    main(['-b', 'cifar10'])
    # main()
    pass
