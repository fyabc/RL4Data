#! /usr/bin/python

from __future__ import print_function, unicode_literals

from functools import wraps
import time
import cPickle as pkl
import numpy as np
from theano import config

from config import Config

__author__ = 'fyabc'

fX = config.floatX


def logging(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print('Start function %s...' % func.__name__, end='')
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        print('%s done, time: %.3fs' % (func.__name__, end_time - start_time))
        return result
    return wrapper


def floatX(value):
    return np.asarray(value, dtype=fX)


def unpickle(filename):
    with open(filename, 'rb') as f:
        return pkl.load(f)


@logging
def load_cifar10_data(data_dir=Config['data_dir'], train_size=Config['train_size']):
    xs = []
    ys = []
    for j in range(5):
        d = unpickle(data_dir + '/data_batch_%d' % j)
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


def test():
    pass


if __name__ == '__main__':
    test()
