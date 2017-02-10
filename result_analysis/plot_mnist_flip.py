#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import matplotlib.pyplot as plt
from utils import init, legend, plot_acc_line, Curves, get_data

__author__ = 'fyabc'

test_acc = get_data('tests_accuracy', 'mnist_flip')
drop_number = get_data('case_numbers', 'mnist_flip')


def plot_all_acc_lines(interval=1, **kwargs):
    use_ac = kwargs.pop('use_ac', True)
    smooth = kwargs.pop('smooth', 300)

    plot_acc_line(Curves[0].title, '-.', drop_number, test_acc, interval, smooth)
    if use_ac:
        plot_acc_line(Curves[1].title, '-.', drop_number, test_acc, interval, smooth)
    plot_acc_line(Curves[2].title, '--', drop_number, test_acc, interval, smooth)
    plot_acc_line(Curves[3].title, '--', drop_number, test_acc, interval, smooth)
    if use_ac:
        plot_acc_line(Curves[4].title, '-', drop_number, test_acc, interval, smooth)
    plot_acc_line(Curves[5].title, '-', drop_number, test_acc, interval, smooth)


def plot_acc():
    init(xmin=600000, xmax=4000000, ymax=0.96, ymin=0.90)
    plot_all_acc_lines(interval=11, use_ac=True, smooth=300)
    legend(use_ac=True)
    plt.show()


if __name__ == '__main__':
    plot_acc()
