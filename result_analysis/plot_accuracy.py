# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import os
import sys

ProjectRootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ProjectRootPath)

from itertools import count

import numpy as np
from scipy.interpolate import spline
import matplotlib.pyplot as plt

from libs.utility.config import LogPath
from utils import Curves, legend, average_list, move_avg, CFG, pick_interval

Interval = 4


def get_data_list(filename, dataset='mnist', interval=Interval, start_tags=('Test accuracy:', 'TeA:')):
    if filename is None:
        return None

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

    min_len = min(len(x), len(y), maxlen)
    x, y = x[:min_len], y[:min_len]

    if mv_avg is not False:
        y = move_avg(y, mv_avg)

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

    x_new = pick_interval(x_new, interval)
    power_smooth = pick_interval(power_smooth, interval)

    plt.plot(x_new, power_smooth, style, label=title, **kwargs)


def plot_for_paper_all(*filenames, **kwargs):
    dataset = kwargs.pop('dataset', 'mnist')
    interval = kwargs.pop('interval', 5)
    vp_size = kwargs.pop('vp_size', 2500)
    maxlen = kwargs.pop('maxlen', 400)
    smooth = kwargs.pop('smooth', 200)
    spl_cfg = kwargs.pop('spl_cfg', [165])
    speed_cfg = kwargs.pop('speed_cfg', ['-'])
    line_width = kwargs.pop('line_width', CFG['linewidth'])
    style = kwargs.pop('style', None)
    legend_loc = kwargs.pop('legend_loc', 'out')
    add_sort = kwargs.pop('add_sort', False)

    spl_count = len(spl_cfg)
    speed_count = len(speed_cfg)

    if style == 'cn':
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # Plot test accuracy
    # plt.title(r'$Test\ Accuracy$', fontsize=40)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=24)

    xmin = kwargs.pop('xmin', 160)
    xmax = kwargs.pop('xmax', 1200)
    ymin = kwargs.pop('ymin', 0.89)
    ymax = kwargs.pop('ymax', 0.96)
    mv_avg = kwargs.pop('mv_avg', False)

    test_acc_lists = get_test_acc_lists(*filenames, interval=1, dataset=dataset)
    raw = test_acc_lists[0]
    spls = test_acc_lists[1:1 + spl_count]
    random_drop = test_acc_lists[1 + spl_count]

    if add_sort:
        raw_sort = test_acc_lists[2 + spl_count]

    reinforces = test_acc_lists[-speed_count:]

    raw_line_style = '-o'
    random_line_style = '-s'
    sorted_line_style = '-*'
    spl_line_style = '--'
    reinforce_line_style = '-'

    # The color should be fixed.
    spl_colors = ['g', 'm', 'y']
    reinforce_colors = ['c', 'm', 'y', 'g', 'k', 'b', 'r', 'p']
    raw_color = 'r'
    random_drop_color = 'b'
    sort_color = 'g'

    title = {
        None: r'$NDF{}$',
        'l2t': r'$L2T{}$',
        'cn': r'$NDF{}$',
    }[style]
    for i, reinforce in enumerate(reinforces):
        if speed_cfg[i] != '':
            speed_cfg[i] = '-' + speed_cfg[i]

        plot_accuracy_curve(title.format(speed_cfg[i]), reinforce_colors[i] + reinforce_line_style,
                            reinforce, vp_size, smooth, interval, maxlen,
                            linewidth=line_width, mv_avg=mv_avg, markersize=CFG['markersize'])

    title = {
        None: r'$SPL{}$',
        'l2t': r'$SPL{}$',
        'cn': r'$SPL{}$',
    }[style]
    for i, spl in enumerate(spls):
        if spl_cfg[i] != '':
            spl_cfg[i] = '-' + spl_cfg[i]
        plot_accuracy_curve(title.format(spl_cfg[i]), spl_colors[i] + spl_line_style,
                            spl, vp_size, smooth, interval, maxlen,
                            linewidth=line_width, mv_avg=mv_avg, markersize=CFG['markersize'])

    if random_drop is not None:
        title = {
            None: Curves[0].title,
            'l2t': '$RandTeach$',
            'cn': Curves[0].title,
        }[style]
        plot_accuracy_curve(title, random_drop_color + random_line_style, random_drop, vp_size, smooth, interval,
                            maxlen, linewidth=line_width - 1, mv_avg=mv_avg, markersize=CFG['markersize'])

    if add_sort:
        title = '$CL$'
        plot_accuracy_curve(title, sort_color + sorted_line_style, raw_sort, vp_size, smooth, interval, maxlen,
                            linewidth=line_width - 1, mv_avg=mv_avg, markersize=CFG['markersize'])

    title = {
        None: Curves[3].title,
        'l2t': '$NoTeach$',
        'cn': Curves[3].title,
    }[style]
    plot_accuracy_curve(title, raw_color + raw_line_style, raw, vp_size, smooth, interval, maxlen,
                        linewidth=line_width - 1, mv_avg=mv_avg, markersize=CFG['markersize'])

    legend(use_ac=False, spl_count=spl_count, speed_count=speed_count, n_rows=3,
           rand_drop_count=random_drop is not None, legend_loc=legend_loc)

    x_label = {
        None: '$Number\ of\ Effective\ Training\ Data$',
        'l2t': '$Number\ of\ Effective\ Training\ Data$',
        'cn': u'有效训练样本数',
    }[style]
    y_label = {
        None: r'$Test\ Accuracy$',
        'l2t': r'$Test\ Accuracy$',
        'cn': u'测试集准确率',
    }[style]

    plt.xlabel(x_label, fontsize=30)
    plt.ylabel(y_label, fontsize=30)

    plt.xlim(xmin=xmin * vp_size, xmax=xmax * vp_size)
    plt.ylim(ymin=ymin, ymax=ymax)
    plt.grid(True, axis='both', linestyle='--')

    # End plot test accuracy

    plt.show()


def plot_for_paper_mnist():
    plot_for_paper_all(
        'log-mnist-raw-NonC1.txt',

        'log-mnist-spl-NonC5.txt',
        'log-mnist-spl-NonC4.txt',
        'log-mnist-spl-NonC6.txt',

        'log-mnist-random_drop-speed-NonC3.txt',

        # 'log-mnist-stochastic-lr-speed-NonC3Best.txt',
        'log-mnist-stochastic-lr-speed-NonC7Best.txt',
        'log-mnist-stochastic-lr-speed-NonC8Best.txt',
        # 'log-mnist-stochastic-lr-speed-NonC14Best.txt',
        'log-mnist-stochastic-lr-speed-NonC10Best.txt',

        # 'log-mnist-stochastic-lr-speed-Cifar10NonC2Best.txt',
        # 'log-mnist-stochastic-lr-speed-Cifar10NonC3Best.txt',
        # 'log-mnist-stochastic-lr-speed-Cifar10NonC4Best.txt',

        # 'log-mnist-stochastic-lr-speed-Label.txt',
        # 'log-mnist-stochastic-lr-speed-NoLoss_1.txt',
        # 'log-mnist-stochastic-lr-speed-NoOutput_1.txt',
        # 'log-mnist-stochastic-lr-speed-NoTrainInfo.txt',

        xmin=0,
        xmax=600,
        ymin=0.93,
        ymax=0.981,
        interval=15,
        maxlen=600,
        smooth=800,

        interval2=200,
        xmax2=200000,
        smooth2=200,
        mv_avg2=20,

        spl_cfg=['80', '120', '160'],

        speed_cfg=['0.94', '0.96', '0.98', ],

        style=None,
    )


def plot_for_paper_mnist_transfer():
    plot_for_paper_all(
        'log-mnist-raw-NonC1.txt',

        'log-mnist-spl-NonC5.txt',
        # 'log-mnist-spl-NonC4.txt',
        # 'log-mnist-spl-NonC6.txt',

        'log-mnist-random_drop-speed-NonC3.txt',

        # 'log-mnist-stochastic-lr-speed-NonC3Best.txt',
        # 'log-mnist-stochastic-lr-speed-NonC7Best.txt',
        # 'log-mnist-stochastic-lr-speed-NonC14Best.txt',
        'log-mnist-stochastic-lr-speed-NonC8Best.txt',
        # 'log-mnist-stochastic-lr-speed-NonC10Best.txt',

        # 'log-mnist-stochastic-lr-speed-Cifar10NonC2Best.txt',
        'log-mnist-stochastic-lr-speed-Cifar10NonC3Best.txt',
        # 'log-mnist-stochastic-lr-speed-Cifar10NonC4Best.txt',

        xmin=0,
        xmax=600,
        ymin=0.93,
        ymax=0.981,
        interval=15,
        maxlen=600,
        smooth=800,

        interval2=200,
        xmax2=200000,
        smooth2=200,
        mv_avg2=20,

        spl_cfg=[''],

        speed_cfg=['', 'Transfer'],

        style=None,

        legend_loc='lower right',
    )


def plot_for_paper_mnist_half():
    plot_for_paper_all(
        'log-mnist-raw-Half.txt',

        'log-mnist-spl-Half_60.txt',
        'log-mnist-spl-Half_120.txt',
        'log-mnist-spl-Half_180.txt',

        # 'log-mnist-random_drop-speed-NonC3.txt',
        None,

        # 'log-mnist-stochastic-lr-speed-NonC3Best.txt',
        'log-mnist-stochastic-lr-speed-Half_MnistNonC7Best.txt',
        'log-mnist-stochastic-lr-speed-Half_MnistNonC14Best.txt',
        'log-mnist-stochastic-lr-speed-Half_MnistNonC10Best.txt',
        # 'log-mnist-stochastic-lr-speed-Half_MnistNonC10Best.txt',

        # 'log-mnist-stochastic-lr-speed-Cifar10NonC2Best.txt',
        # 'log-mnist-stochastic-lr-speed-Cifar10NonC3Best.txt',
        # 'log-mnist-stochastic-lr-speed-Cifar10NonC4Best.txt',

        # 'log-mnist-stochastic-lr-speed-Label.txt',
        # 'log-mnist-stochastic-lr-speed-NoLoss_1.txt',
        # 'log-mnist-stochastic-lr-speed-NoOutput_1.txt',
        # 'log-mnist-stochastic-lr-speed-NoTrainInfo.txt',

        xmin=50,
        xmax=600,
        ymin=0.92,
        ymax=0.974,
        interval=15,
        maxlen=600,
        smooth=800,

        interval2=200,
        xmax2=200000,
        smooth2=200,
        mv_avg2=20,

        # spl_cfg=[80, 120, 160],
        spl_cfg=['80', '120', '160'],

        # speed_cfg=['0.94', '.89\ .92\ .94', '.80\ .88\ .96'],
        speed_cfg=['0.93', '0.95', '0.97',
                   # '-C0.80', '-C0.84', '-C0.88'],
                   ],
        # 'Label', 'NoLoss', 'NoOutput', 'NoTrainInfo'],
        # speed_cfg=['', '-C'],

        style='cn',

        legend_loc='lower right',
    )


def plot_for_paper_mnist_half_feature_analysis():
    plot_for_paper_all(
        'log-mnist-raw-Half.txt',

        # 'log-mnist-spl-Half_60.txt',
        # 'log-mnist-spl-Half_120.txt',
        # 'log-mnist-spl-Half_180.txt',

        # 'log-mnist-random_drop-speed-NonC3.txt',
        None,

        'log-mnist-stochastic-lr-speed-Half_MnistNonC8Best.txt',
        'log-mnist-stochastic-lr-speed-NoLoss_Half_1.txt',
        'log-mnist-stochastic-lr-speed-NoTrainInfo_Half.txt',
        'log-mnist-stochastic-lr-speed-Half_MnistNonC10Best.txt',

        xmin=80,
        xmax=600,
        ymin=0.93,
        ymax=0.976,
        interval=15,
        maxlen=600,
        smooth=800,

        interval2=200,
        xmax2=200000,
        smooth2=200,
        mv_avg2=20,

        # spl_cfg=[80, 120, 160],
        spl_cfg=[],

        # speed_cfg=['0.94', '.89\ .92\ .94', '.80\ .88\ .96'],
        speed_cfg=[
            'NoDataFeatures',
            'NoCombinedFeatures',
            'NoModelFeatures',
            'AllFeatures',
        ],

        style='cn',

        legend_loc='lower right',
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

        # 'log-cifar10-stochastic-lr-speed-MnistNonC7Best.txt',
        # 'log-cifar10-stochastic-lr-speed-MnistNonC8Best.txt',
        # 'log-cifar10-stochastic-lr-speed-MnistNonC10Best.txt',
        # 'log-cifar10-raw-LoadIndex_32.txt',
        # 'log-cifar10-stochastic-lr-speed-DumpIndex_32.txt',

        dataset='cifar10',
        xmin=0,
        xmax=89,
        ymin=0.63,
        ymax=0.94,
        interval=10,
        vp_size=390 * 128,
        smooth=800,
        mv_avg=2,

        ymin2=None,
        ymax2=None,
        xmax2=40000,

        # spl_cfg=[120, 60, 180],
        spl_cfg=['120', '60', '180'],

        # speed_cfg=['.80\ .84\ .865', '.84', '.80'],
        speed_cfg=['0.80', '0.84', '0.88',
                   # '-M0.94', '-M0.96', '-M0.98'],
                   ],
        # speed_cfg=['', '-M'],

        style='cn',

        legend_loc='lower right'
    )


def plot_for_paper_cifar_half():
    plot_for_paper_all(
        'log-cifar10-raw-ResNet32_Half.txt',

        'log-cifar10-spl-ResNet32_Half_210.txt',
        'log-cifar10-spl-ResNet32_Half_2.txt',
        'log-cifar10-spl-ResNet32_Half_180_1.txt',

        'log-cifar10-random_drop-lr-speed-Cifar10NonC2Best_Half.txt',

        'log-cifar10-stochastic-lr-speed-Cifar10NonC2Best_Half.txt',
        'log-cifar10-stochastic-lr-speed-Cifar10NonC3Best_Half.txt',
        'log-cifar10-stochastic-lr-speed-Cifar10NonC4Best_Half.txt',
        # 'log-cifar10-stochastic-lr-speed-NonC3_Half.txt',

        dataset='cifar10',
        xmin=0,
        xmax=91,
        ymin=0.68,
        ymax=0.91,
        interval=10,
        vp_size=390 * 128,
        smooth=800,
        mv_avg=2,

        ymin2=None,
        ymax2=None,
        xmax2=40000,

        spl_cfg=['120', '150', '180'],

        # speed_cfg=['.80\ .84\ .865', '.84', '.80'],
        # speed_cfg=['0.80', '0.84', '0.88',
        # '-M0.94', '-M0.96', '-M0.98'],
        speed_cfg=['0.80', '0.84', '0.88'],
        # speed_cfg=['', '-M'],

        style='cn',

        legend_loc='lower right'
    )


def plot_for_paper_cifar_transfer():
    plot_for_paper_all(
        'log-cifar10-raw-NonC1.txt',

        # 'log-cifar10-spl-NonC1.txt',
        # 'log-cifar10-spl-NonC60.txt',
        'log-cifar10-spl-NonC180.txt',

        'log-cifar10-random_drop-speed-NonC2.txt',

        # 'log-cifar10-stochastic-lr-speed-NonC2Best_1.txt',
        'log-cifar10-stochastic-lr-speed-NonC3Best.txt',
        # 'log-cifar10-stochastic-lr-speed-NonC4Best.txt',

        'log-cifar10-stochastic-lr-speed-MnistNonC10Best.txt',

        dataset='cifar10',
        xmin=0,
        xmax=89,
        ymin=0.63,
        ymax=0.94,
        interval=10,
        vp_size=390 * 128,
        smooth=800,
        mv_avg=2,

        ymin2=None,
        ymax2=None,
        xmax2=40000,

        spl_cfg=[''],

        speed_cfg=['', 'Transfer'],
        # speed_cfg=['Transfer'],

        style='cn',

        legend_loc='lower right',
    )


def plot_for_paper_cifar_resnet110():
    plot_for_paper_all(
        'log-cifar10-raw-ResNet110.txt',

        'log-cifar10-spl-ResNet110.txt',
        # 'log-cifar10-spl-ResNet110_180.txt',
        # 'log-cifar10-spl-ResNet110_60.txt',

        'log-cifar10-random_drop-lr-speed-Cifar10NonC3Best_Resnet110.txt',

        'log-cifar10-stochastic-lr-speed-Cifar10NonC3Best_Resnet110.txt',
        # 'log-cifar10-stochastic-lr-speed-Cifar10NonC2Best_Resnet110.txt',
        # 'log-cifar10-raw-LoadIndex_110.txt',

        dataset='cifar10',
        xmin=0,
        xmax=91,
        ymin=0.65,
        ymax=0.93,
        interval=10,
        vp_size=390 * 128,
        smooth=800,
        mv_avg=2,

        ymin2=None,
        ymax2=None,
        xmax2=40000,

        spl_cfg=['',
                 # '180', '60'],
                 ],

        # speed_cfg=['.80\ .84\ .865', '.84', '.80'],
        # speed_cfg=['0.80-ResNet32', '0.84-ResNet32', 'DataPath-Resnet32'],
        speed_cfg=['Transfer'],

        style='cn',

        legend_loc='lower right',
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
        'log-imdb-raw-NonC1s.txt',

        # 'log-imdb-spl-NonC1.txt',
        'log-imdb-spl-NonC2.txt',
        'log-imdb-spl-NonC100.txt',
        'log-imdb-spl-NonC120.txt',

        'log-imdb-random_drop-lr-speed-NonC_old2_2.txt',

        'log-imdb-stochastic-lr-speed-NonC_Old2_1.txt',
        'log-imdb-stochastic-lr-speed-OldNonC2Warm2.txt',
        'log-imdb-stochastic-lr-speed-NonC_Old2_2.txt',
        # 'log-imdb-stochastic-lr-speed-NonC_Old.txt',

        # 'log-imdb-stochastic-mlp-speed-NonC1Best.txt',
        # 'log-imdb-stochastic-mlp-speed-NonC2Best.txt',

        dataset='imdb',
        xmin=0,
        xmax=27.5 * 7 / 6,
        # xmax=90,
        ymin=0.45,
        ymax=0.90,
        interval=5,
        vp_size=200 * 16,
        smooth=800,
        mv_avg=2,

        ymin2=None,
        ymax2=None,
        xmax2=40000,
        interval2=200,
        smooth2=800,
        mv_avg2=5,

        # speed_cfg=['.80\ .83\ .86', '.70\ .80\ .85', '.80\ .83\ .86'],
        # speed_cfg=['0.80', '0.83', '0.86'],
        speed_cfg=['-0.80', '-0.83', '-0.86'],

        # spl_cfg=[80, 100, 120],
        spl_cfg=['-80', '-100', '-120'],

        style='l2t',
    )


def plot_for_paper_imdb_half():
    plot_for_paper_all(
        'log-imdb-raw-Half.txt',

        'log-imdb-spl-Half_80.txt',
        'log-imdb-spl-Half_100.txt',
        'log-imdb-spl-Half_120.txt',

        'log-imdb-random_drop-lr-speed-Half_NonC_Old2.txt',

        'log-imdb-raw-Half_sort.txt',

        'log-imdb-stochastic-lr-speed-Half_NonC_Old2.txt',
        'log-imdb-stochastic-lr-speed-Half_OldNonC2Warm2.txt',
        'log-imdb-stochastic-lr-speed-Half_NonC_Old2_1.txt',

        dataset='imdb',
        xmin=1,
        xmax=27.5 * 7 / 6,
        ymin=0.47,
        ymax=0.86,
        interval=1,
        vp_size=200 * 16,
        smooth=200,
        mv_avg=3,

        # speed_cfg=['.80\ .83\ .86', '.70\ .80\ .85', '.80\ .83\ .86'],
        speed_cfg=['0.78', '0.81', '0.84'],

        spl_cfg=['80', '100', '120'],

        add_sort=True,

        style='cn',

        legend_loc='upper left',
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
            'mnist-half': plot_for_paper_mnist_half,
            'mnist-transfer': plot_for_paper_mnist_transfer,
            'feature-analysis': plot_for_paper_mnist_half_feature_analysis,
            'c-mnist': plot_for_paper_c_mnist,
            'cifar10': plot_for_paper_cifar,
            'cifar10-transfer': plot_for_paper_cifar_transfer,
            'cifar10-half': plot_for_paper_cifar_half,
            'c-cifar10': plot_for_paper_c_cifar,
            'cifar10-resnet110': plot_for_paper_cifar_resnet110,
            'imdb': plot_for_paper_imdb,
            'imdb-half': plot_for_paper_imdb_half,
        }[options.builtin]()


if __name__ == '__main__':
    # main(['-b', 'cifar10-half'])
    # main(['-b', 'cifar10-resnet110'])
    # main(['-b', 'feature-analysis'])
    # main(['-b', 'mnist-transfer'])
    main(['-b', 'cifar10-transfer'])
    # main()
    pass
