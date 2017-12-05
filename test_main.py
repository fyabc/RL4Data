#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Main entry for some test functions."""

from libs.utility.plot_utils import simple_plot_test_acc


def plot_test_acc():
    simple_plot_test_acc(
        'log-imdb-raw-new_adam.txt',
        'log-imdb-raw-new.txt',
        dataset='imdb',
        # xmax=20,
    )


def main():
    plot_test_acc()


if __name__ == '__main__':
    main()
