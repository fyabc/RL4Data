# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import os

import numpy as np

from config import CifarConfig
from utils import logging, unpickle, floatX, message, fX

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


def prepare_CIFAR10_data(inputs, targets):
    batch_size = len(targets)
    padded = np.pad(inputs, ((0, 0), (0, 0), (4, 4), (4, 4)), mode=str('constant'))
    random_cropped = np.zeros_like(inputs, dtype=fX)
    crops = np.random.random_integers(0, high=8, size=(batch_size, 2))

    for r in range(batch_size):
        random_cropped[r, :, :, :] = padded[r, :, crops[r, 0]:(crops[r, 0] + 32), crops[r, 1]:(crops[r, 1] + 32)]

    return random_cropped, targets


def iterate_minibatches(inputs, targets, batch_size, shuffle=False, augment=False, return_indices=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        if augment:
            # as in paper :
            # pad feature arrays with 4 pixels on each side
            # and do random cropping of 32x32
            padded = np.pad(inputs[excerpt], ((0, 0), (0, 0), (4, 4), (4, 4)), mode=str('constant'))
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=fX)
            crops = np.random.random_integers(0, high=8, size=(batch_size, 2))

            for r in range(batch_size):
                random_cropped[r, :, :, :] = padded[r, :, crops[r, 0]:(crops[r, 0] + 32),
                                                    crops[r, 1]:(crops[r, 1] + 32)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        if return_indices:
            yield inp_exc, targets[excerpt], excerpt
        else:
            yield inp_exc, targets[excerpt]


def pre_process_CIFAR10_data():
    # Load the dataset
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_cifar10_data(load_cifar10_data())

    train_size, validate_size, test_size = y_train.shape[0], y_validate.shape[0], y_test.shape[0]

    message('Training data size:', train_size)
    message('Validation data size:', validate_size)
    message('Test data size:', test_size)

    return x_train, y_train, x_validate, y_validate, x_test, y_test, train_size, validate_size, test_size
