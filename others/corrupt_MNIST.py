#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import cPickle as pkl
import gzip
import numpy as np
import random

orig_filename = '../data/mnist/mnist.pkl.gz'
flip_filename = '../data/mnist/mnist_corrupted.pkl.gz'
part_flip_filename = '../data/mnist/mnist_corrupted_part.pkl.gz'
part_flip_filename_mid = '../data/mnist/mnist_corrupted_part_mid.pkl.gz'
gaussian_filename = '../data/mnist/mnist_gaussian.pkl.gz'
part_gaussian_filename = '../data/mnist/mnist_gaussian_part.pkl.gz'
gaussian_filename_1 = '../data/mnist/mnist_gaussian_std1.pkl.gz'
part_gaussian_filename_1 = '../data/mnist/mnist_gaussian_std1_part.pkl.gz'


def get_flip():
    with gzip.open(orig_filename, 'rb') as orig_f:
        train, valid, test = pkl.load(orig_f)

    train_x, train_y = train

    # train_x: (50000, 784)
    train_len = train_x.shape[0]
    part_len = train_len // 10

    population = list(range(784))

    for i in range(10):
        k = 784 * i // 10

        for j in range(i * part_len, (i + 1) * part_len):
            data_point = train_x[j]

            for loc in random.sample(population, k):
                data_point[loc] = 1.0 - data_point[loc]

    train = train_x, train_y

    with gzip.open(flip_filename, 'wb') as new_f:
        pkl.dump((train, valid, test), new_f)


def get_part_flip(old_fn=flip_filename, new_fn=part_flip_filename, part_size=range(35000)):
    with gzip.open(old_fn, 'rb') as orig_f:
        train, valid, test = pkl.load(orig_f)

    train_x, train_y = train
    train_x = train_x[part_size]
    train_y = train_y[part_size]

    train = train_x, train_y

    with gzip.open(new_fn, 'wb') as new_f:
        pkl.dump((train, valid, test), new_f)


def get_gaussian(g_fn=gaussian_filename, max_std=0.4):
    with gzip.open(orig_filename, 'rb') as orig_f:
        train, valid, test = pkl.load(orig_f)

    train_x, train_y = train

    # train_x: (50000, 784)
    train_len = train_x.shape[0]
    part_len = train_len // 10

    for i in range(10):
        if i == 0:
            continue

        std = max_std * i / 10

        for j in range(i * part_len, (i + 1) * part_len):
            data_point = train_x[j]

            data_point += np.random.normal(0.0, std, data_point.shape)
            data_point[data_point >= 1.] = 1.
            data_point[data_point <= 0.] = 0.

    train = train_x, train_y

    with gzip.open(g_fn, 'wb') as new_f:
        pkl.dump((train, valid, test), new_f)


def main():
    # get_flip()

    # get_gaussian(gaussian_filename_1, 1.0)
    get_part_flip(flip_filename, part_flip_filename_mid, part_size=np.concatenate((range(20000), range(30000, 50000))))
    pass


if __name__ == '__main__':
    main()
