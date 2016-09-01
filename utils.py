#! /usr/bin/python

from __future__ import print_function, unicode_literals

import sys
import os
from functools import wraps
import random
import time
import cPickle as pkl
import numpy as np
# from theano import config

from config import Config, ParamConfig

__author__ = 'fyabc'

# fX = config.floatX
fX = Config['floatX']

logging_file = sys.stderr

_depth = 0


def logging(func, file_=sys.stderr):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _depth

        print(' ' * 2 * _depth + '[Start function %s...]' % func.__name__, file=file_)
        _depth += 1
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        _depth -= 1
        print(' ' * 2 * _depth + '[Function %s done, time: %.3fs]' % (func.__name__, end_time - start_time), file=file_)
        return result
    return wrapper


def message(*args, **kwargs):
    print(*args, file=logging_file, **kwargs)


def floatX(value):
    return np.asarray(value, dtype=fX)


def init_norm(*dims):
    return floatX(np.random.randn(*dims))


def unpickle(filename):
    with open(filename, 'rb') as f:
        return pkl.load(f)


@logging
def load_cifar10_data(data_dir=Config['data_dir'], train_size=Config['train_size']):
    if not os.path.exists(data_dir):
        raise Exception("CIFAR-10 dataset can not be found. Please download the dataset from "
                        "'https://www.cs.toronto.edu/~kriz/cifar.html'.")

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


def get_small_train_data(x_train, y_train, train_small_size=ParamConfig['train_epoch_size']):
    train_size = x_train.shape[0]

    # Use small dataset to check the code
    sampled_indices = random.sample(range(train_size), train_small_size)
    return x_train[sampled_indices], y_train[sampled_indices]


def shuffle_data(x_train, y_train, train_small_size=ParamConfig['train_epoch_size']):
    shuffled_indices = np.arange(train_small_size)
    np.random.shuffle(shuffled_indices)
    return x_train[shuffled_indices], y_train[shuffled_indices]


# ############################# Batch iterator ###############################
def iterate_minibatches(inputs, targets, batch_size, shuffle=False, augment=False):
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
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0, high=8, size=(batch_size, 2))

            for r in range(batch_size):
                random_cropped[r, :, :, :] = padded[r, :, crops[r, 0]:(crops[r, 0] + 32),
                                                    crops[r, 1]:(crops[r, 1] + 32)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc, targets[excerpt]


def simple_parse_args(args):
    args_dict = {}

    for arg in args:
        if '=' in arg:
            key, value = arg.split('=')

            args_dict[key] = eval(value)

    return args_dict


class CifarDataAnalyzer(object):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.class_num = 10

        self.data_size = self.inputs.shape[0]

    def get_label_distribution(self):
        result = np.zeros((self.class_num,))

        for target in self.targets:
            result[target] += 1
        return result

    def save(self, filename):
        np.savez(filename, self.inputs, self.targets)


def test():
    pass


if __name__ == '__main__':
    test()
