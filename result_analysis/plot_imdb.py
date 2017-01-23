#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import matplotlib.pyplot as plt
from utils import init, legend, plot_acc_line, Curves, get_data

__author__ = 'fyabc'

test_acc = get_data('tests_accuracy', 'imdb')
drop_number = get_data('case_numbers', 'imdb')


def plot_all_acc_lines():
    plot_acc_line(Curves[0].title, '-.', drop_number, test_acc)
    plot_acc_line(Curves[1].title, '-.', drop_number, test_acc)
    plot_acc_line(Curves[2].title, '--', drop_number, test_acc)
    plot_acc_line(Curves[3].title, '--', drop_number, test_acc)
    plot_acc_line(Curves[4].title, '-', drop_number, test_acc)
    plot_acc_line(Curves[5].title, '-', drop_number, test_acc)


def plot_acc():
    init(xmin=0, xmax=120000, ymax=0.9, ymin=0.45)
    plot_all_acc_lines()
    legend()
    plt.show()


if __name__ == '__main__':
    plot_acc()
