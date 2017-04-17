#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import cPickle as pkl
import random
import matplotlib.pyplot as plt
import numpy as np
import gzip


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


def get_gaussian(x, max_std=1.0):
    train_len = x.shape[0]
    part_len = train_len // 10

    for i in range(10):
        if i == 0:
            continue

        std = max_std * i / 10

        for j in range(i * part_len, (i + 1) * part_len):
            data_point = x[j]

            data_point += np.random.normal(0.0, std, data_point.shape)
            data_point[data_point >= 1.] = 1.
            data_point[data_point <= 0.] = 0.

    return x


def save_data(filename, x_train, y_train, x_test, y_test):
    with gzip.open(filename, 'wb') as f:
        pkl.dump(((x_train, y_train), (x_test, y_test)), f)


def create_new(new_fn, x, y, func=get_flip):
    train_size = 50000
    part_size = train_size // 10

    x_t, x_te = x[:train_size], x[train_size:]
    y_t, y_te = y[:train_size], y[train_size:]

    x_t_new = func(x_t)

    save_data(new_fn, x_t_new, y_t, x_te, y_te)


raw_filename = '../data/cifar10/cifar10.pkl.gz'
flip_filename = '../data/cifar10/cifar10_flip.pkl.gz'
gaussian_std05_filename = '../data/cifar10/cifar10_gaussian_std05.pkl.gz'


def main():
    x, y = get_data()

    # train_size = 50000
    # part_size = train_size // 10
    #
    # x_t, x_te = x[:train_size], x[train_size:]
    # y_t, y_te = y[:train_size], y[train_size:]

    # x_t_new = get_gaussian(x_t.copy(), max_std=0.5)

    # plot_100(np.concatenate([
    #     x_t_new[part_size * i: part_size * i + 10]
    #     for i in range(10)
    # ]))

    # create_new(flip_filename, x, y)
    # create_new(gaussian_std05_filename, x, y, func=lambda x: get_gaussian(x, max_std=1.0))
    create_new(raw_filename, x, y, func=lambda x: x)


if __name__ == '__main__':
    main()
