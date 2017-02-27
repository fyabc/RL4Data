#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import cPickle as pkl
import os
from collections import namedtuple
from itertools import izip_longest

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import spline

__author__ = 'fyabc'

DataSet = 'imdb'

Curve = namedtuple('Curve', ['raw_name', 'title'])
Curves = [
    Curve('random_drop_speed', r'$RandDrop$'),
    Curve('random_drop_AC', r'$RandDropActorCritic$'),
    Curve('SPL', r'$SPL$'),
    Curve('raw', r'$UnfilteredSGD$'),
    Curve('AC_valacc', r'$NDF-ActorCritic$'),
    Curve('valacc', r'$NDF$'),
]


CFG = {
    'linewidth': 7.5,
    'markersize': 17.0,
}


def load_list(filename, dtype=float):
    if not os.path.exists(filename):
        return []

    with open(filename) as f:
        return map(lambda l: dtype(l.strip()), f)


def save_list(l, filename):
    with open(filename, 'w') as f:
        for i in l:
            f.write(str(i) + '\n')


def avg(l):
    return sum(l) / len(l)


def average_without_none(l):
    no_none = [e for e in l if e is not None]

    return sum(no_none) / len(no_none)


def average_list(*lists):
    return [
        average_without_none(elements)
        for elements in izip_longest(*lists)
    ]


def move_avg(l, mv_avg=5):
    return [avg(l[max(i - mv_avg, 0):i + 1]) for i in range(len(l))]


def pick_interval(l, interval=1):
    if interval <= 1:
        return l

    return [e for i, e in enumerate(l) if (i + 1) % interval == 0]


def get_data(data_field, dataset=DataSet):
    dumped_filename = './dumped/{}/{}.pkl'.format(dataset, data_field)

    if not os.path.exists(dumped_filename):
        data = {
            curve.title: load_list('./data/{}/{}_{}.txt'.format(dataset, curve.raw_name, data_field))
            for curve in Curves
        }
        with open(dumped_filename, 'wb') as dumped_file:
            pkl.dump(data, dumped_file)
        return data
    else:
        with open(dumped_filename, 'rb') as dumped_file:
            return pkl.load(dumped_file)


def init(**kwargs):
    ax = plt.gca()
    ax.yaxis.grid()
    plt.ylim(ymin=kwargs.pop('ymin', None), ymax=kwargs.pop('ymax', None))
    plt.xlim(xmin=kwargs.pop('xmin', None), xmax=kwargs.pop('xmax', None))
    plt.xlabel(r'$Number\ of\ Training\ Instances$', fontsize=30, fontweight='bold')
    plt.ylabel(r'$Test\ Accuracy$', fontsize=30, fontweight='bold')


def legend(**kwargs):
    use_ac = kwargs.pop('use_ac', True)
    spl_count = kwargs.pop('spl_count', 1)
    speed_count = kwargs.pop('speed_count', 1)
    n_rows = kwargs.pop('n_rows', 2)

    total_number = 1 + 1 + spl_count + speed_count

    if use_ac:
        total_number += 2

    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.0), bbox_transform=plt.gcf().transFigure,
               fancybox=False, shadow=False,
               ncol=total_number // n_rows + int(bool(total_number % n_rows)), fontsize=28,
               borderpad=0.2, labelspacing=0.2, handletextpad=0.2, borderaxespad=0.2)


def plot_acc_line(k, style, xs, ys, interval=1, smooth=300):
    dx, dy = xs[k], ys[k]

    if interval > 1:
        dx = [e for i, e in enumerate(dx) if (i + 1) % interval == 0]
        dy = [e for i, e in enumerate(dy) if (i + 1) % interval == 0]

    min_len = min(len(dx), len(dy))
    dx, dy = dx[:min_len], dy[:min_len]

    x_new = np.linspace(min(dx), max(dx), smooth)
    power_smooth = spline(dx, dy, x_new)

    plt.plot(x_new, power_smooth, linewidth=2.0, label=k, linestyle=style)


ModelPath = './model'


def get_MLP_model(filename):
    abs_filename = os.path.join(ModelPath, filename)

    with open(abs_filename, 'r') as f:
        lines = f.read().split('$')

        W0 = np.matrix(
            str(lines[1][9:].replace('[', '').replace(']\n', ';')[:-1]),
            dtype='float32')
        b0 = np.fromstring(lines[2][9:].replace('[', '').replace(']', ''), sep=' ', dtype='float32')
        W1 = np.fromstring(lines[3][9:].replace('[', '').replace(']', ''), sep=' ', dtype='float32')
        b1 = np.float32(float(lines[4][9:]))

        print(W0, b0, W1, b1, sep='\n')

        np.savez(os.path.splitext(abs_filename)[0] + '.npz', W0, b0, W1, b1)
