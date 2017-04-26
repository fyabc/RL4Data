#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import matplotlib.pyplot as plt
from utils import init, legend, plot_acc_line, Curves, get_data

test_acc = get_data('tests_accuracy', 'imdb')
drop_number = get_data('case_numbers', 'imdb')


def plot_all_acc_lines(**kwargs):
    use_ac = kwargs.pop('use_ac', True)
    smooth = kwargs.pop('smooth', 300)
    interval = kwargs.pop('kwargs', 1)

    plot_acc_line(Curves[0].title, '-.', drop_number, test_acc, interval, smooth)
    if use_ac:
        plot_acc_line(Curves[1].title, '-.', drop_number, test_acc, interval, smooth)
    plot_acc_line(Curves[2].title, '--', drop_number, test_acc, interval, smooth)
    plot_acc_line(Curves[3].title, '--', drop_number, test_acc, interval, smooth)
    if use_ac:
        plot_acc_line(Curves[4].title, '-', drop_number, test_acc, interval, smooth)
    plot_acc_line(Curves[5].title, '-', drop_number, test_acc, interval, smooth)


def plot_acc():
    init(xmin=0, xmax=120000, ymax=0.9, ymin=0.45)
    plot_all_acc_lines(use_ac=False, smooth=250)
    legend(use_ac=False)
    plt.show()


if __name__ == '__main__':
    plot_acc()
