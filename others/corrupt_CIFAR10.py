#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import cPickle as pkl
import random
import matplotlib.pyplot as plt
import numpy as np


def unpickle(filename):
    with open(filename, 'rb') as f:
        return pkl.load(f)


def get_data(data_dir='../data/cifar-10-batches-py'):
    xs = []
    ys = []
    for j in range(5):
        d = unpickle(data_dir + '/data_batch_%d' % (j + 1))
        xs.append(d['data'])
        ys.append(d['labels'])

    d = unpickle(data_dir + '/test_batch')
    xs.append(d['data'])
    ys.append(d['labels'])

    x = np.concatenate(xs) / np.float32(255)
    y = np.concatenate(ys)

    # x: (60000, 3072)  data-type: float32 (0.0 ~ 1.0)
    # y: (60000,)       data-type: int8 (0 ~ 9)

    return x, y


def plot_image(data):
    img = plt.imshow(data.reshape((3, 32, 32)).transpose(1, 2, 0), interpolation=None)
    plt.show()


def plot_100(data):
    for i in range(10):
        for j in range(10):
            plt.subplot(10, 10, 10 * i + j + 1)
            plt.imshow(data[i * 10 + j].reshape((3, 32, 32)).transpose(1, 2, 0), interpolation=None)
    plt.show()


def plot_100_classified(data, label):
    for i in range(10):
        candidates = np.argwhere(label == i)

        for j in range(min(10, len(candidates))):
            plt.subplot(10, 10, 10 * i + j + 1)
            plt.imshow(data[candidates[j][0]].reshape((3, 32, 32)).transpose(1, 2, 0), interpolation=None)

    plt.show()


def get_flip(x):
    train_len = x.shape[0]
    part_len = train_len // 10

    population = list(range(3072))

    for i in range(10):
        k = 3072 * i // 10

        for j in range(i * part_len, (i + 1) * part_len):
            data_point = x[j]

            for loc in random.sample(population, k):
                data_point[loc] = 1.0 - data_point[loc]

    return x


def main():
    x, y = get_data()

    train_size = 5000
    part_size = train_size // 10

    x_t, x_te = x[:train_size], x[50000:]
    y_t, y_te = y[:train_size], y[50000:]

    x_t_new = get_flip(x_t.copy())

    plot_100(np.concatenate([
        x_t_new[part_size * i: part_size * i + 10]
        for i in range(10)
    ]))


if __name__ == '__main__':
    main()
