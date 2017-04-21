#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import os
import gzip
import cPickle as pkl
import matplotlib.pyplot as plt

__author__ = 'fyabc'

DataPath = 'C:/Users/v-yanfa/PycharmProjects/PG4NN_cifar/data'


PlotConfig = {
    'mnist': {
        'shape': (28, 28),
        'cmap': 'gray',
    },
    'cifar10': {
        'shape': (3, 32, 32),
        'cmap': None,
    }
}


def plot_images(dataset, filename, n_class=10, n_sample=10):
    config = PlotConfig[dataset]

    with gzip.open(os.path.join(DataPath, dataset, filename), 'rb') as f:
        train = pkl.load(f)[0]
        train_x, _ = train

    size = len(train_x)

    for i in range(n_class):
        for j in range(n_sample):
            plt.subplot(n_class, n_sample, n_sample * i + j + 1)
            data = train_x[size // n_class * i + j].reshape(config['shape'])

            if dataset == 'cifar10':
                data = data.transpose(1, 2, 0)

            plt.imshow(data,
                       cmap=config['cmap'],
                       interpolation='none')

            plt.axis('off')
    plt.show()


if __name__ == '__main__':
    plot_images('mnist', 'mnist_corrupted.pkl.gz')
    # plot_images('cifar10', 'cifar10_flip.pkl.gz')
