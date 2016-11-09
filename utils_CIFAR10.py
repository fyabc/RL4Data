# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import os

import numpy as np

from config import CifarConfig
from utils import logging, unpickle, floatX

__author__ = 'fyabc'


@logging
def load_cifar10_data(data_dir=CifarConfig['data_dir']):
    """

    Args:
        data_dir: directory of the CIFAR-10 data.

    Returns:
        A dict, which contains train data and test data.
        Shapes of data:
            x_train:    (100000, 3, 32, 32)
            y_train:    (100000,)
            x_test:     (10000, 3, 32, 32)
            y_test:     (10000,)
    """

    if not os.path.exists(data_dir):
        raise Exception("CIFAR-10 dataset can not be found. Please download the dataset from "
                        "'https://www.cs.toronto.edu/~kriz/cifar.html'.")

    train_size = CifarConfig['train_size']

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
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0, 3, 1, 2)

    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:train_size], axis=0)
    # pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
    x -= pixel_mean

    # create mirrored images
    x_train = x[0:train_size, :, :, :]
    y_train = y[0:train_size]
    x_train_flip = x_train[:, :, :, ::-1]
    y_train_flip = y_train
    x_train = np.concatenate((x_train, x_train_flip), axis=0)
    y_train = np.concatenate((y_train, y_train_flip), axis=0)

    x_test = x[train_size:, :, :, :]
    y_test = y[train_size:]

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    return {
        'x_train': floatX(x_train),
        'y_train': y_train.astype('int32'),
        'x_test': floatX(x_test),
        'y_test': y_test.astype('int32')
    }


def split_cifar10_data(data):
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    # One: validate is not part of train
    x_validate = x_test[:CifarConfig['validation_size']]
    y_validate = y_test[:CifarConfig['validation_size']]
    x_test = x_test[-CifarConfig['test_size']:]
    y_test = y_test[-CifarConfig['test_size']:]

    # # Another: validate is part of train
    # x_validate = x_train[:CifarConfig['validation_size']]
    # y_validate = y_train[:CifarConfig['validation_size']]

    return x_train, y_train, x_validate, y_validate, x_test, y_test
