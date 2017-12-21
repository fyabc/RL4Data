#! /usr/bin/python
# -*- coding: utf-8 -*-

from itertools import count, izip_longest
import os

import matplotlib.pyplot as plt

from .config import LogPath

Interval = 4


def avg(l):
    return sum(l) / len(l)


def average_without_none(l):
    no_none = [e for e in l if e is not None]

    if not no_none:
        return None

    return sum(no_none) / len(no_none)


def average_list(*lists):
    return [
        average_without_none(elements)
        for elements in izip_longest(*lists)
    ]


def move_avg(l, mv_avg=5):
    return [average_without_none(l[max(i - mv_avg, 0):i + 1]) for i in range(len(l))]


def get_file_config(filename, dataset='mnist'):
    abs_filename = os.path.join(LogPath, dataset, filename)

    with open(abs_filename, 'r') as f:
        config_lines = []
        start_tag = False
        for line in f:
            if line.startswith('[Message before train done]'):
                break

            if start_tag:
                config_lines.append(line)

            if not start_tag and line.startswith('The configures and hyperparameters are:'):
                start_tag = True

    config = eval(''.join(config_lines).replace('\n', ' '))
    return config


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


def simple_plot_test_acc(*filenames, **kwargs):
    dataset = kwargs.pop('dataset', 'mnist')

    configs = [get_file_config(filename, dataset=dataset) for filename in filenames]
    vp_size = configs[0][dataset]['valid_freq']

    assert all(config[dataset]['valid_freq'] == vp_size for config in configs), 'Validation point size must be same'

    test_acc_lists = get_test_acc_lists(*filenames, interval=1, dataset=dataset)

    for filename, test_acc_list in zip(filenames, test_acc_lists):
        name = os.path.splitext(filename)[0]
        plt.plot(test_acc_list, label=name)
        print '{} Max = {:.6f}'.format(name, max(test_acc_list))

    plt.title('Test Accuracy')
    plt.grid()
    plt.xlabel('Validation Point (size = {})'.format(vp_size))
    plt.ylabel('Accuracy%')
    plt.xlim(xmax=kwargs.pop('xmax', None), xmin=kwargs.pop('xmin', None))
    plt.legend(loc='best')
    plt.show()
