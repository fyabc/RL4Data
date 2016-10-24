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

from config import Config, CifarConfig, PolicyConfig

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


def average(sequence):
    if sequence is None:
        return 0.0
    if len(sequence) == 0:
        return 0.0
    return sum(sequence) / len(sequence)


###############################
# Data loading and processing #
###############################

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


def get_part_data(x_data, y_data, part_size=None):
    if part_size is None:
        return x_data, y_data

    train_size = x_data.shape[0]
    if train_size < part_size:
        return x_data, y_data

    # Use small dataset to check the code
    sampled_indices = random.sample(range(train_size), part_size)
    return x_data[sampled_indices], y_data[sampled_indices]


def shuffle_data(x_train, y_train):
    shuffled_indices = np.arange(y_train.shape[0])
    np.random.shuffle(shuffled_indices)
    return x_train[shuffled_indices], y_train[shuffled_indices]


##############################
# Other utilities of CIFAR10 #
##############################

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


########################################
# Simple command line arguments parser #
########################################

def simple_parse_args(args, param_config=CifarConfig):
    args_dict = {}
    policy_args_dict = {}
    param_args_dict = {}

    for arg in args:
        arg = arg.replace('@', '"')

        if '=' in arg:
            if arg.startswith('G.'):
                arg = arg[2:]
                the_dict = args_dict
                target_dict = Config
            elif arg.startswith('P.'):
                arg = arg[2:]
                the_dict = policy_args_dict
                target_dict = PolicyConfig
            else:
                the_dict = param_args_dict
                target_dict = param_config
            key, value = arg.split('=')
            if key not in target_dict:
                raise Exception('The key {} is not in the parameters.'.format(key))

            the_dict[key] = eval(value)

    return args_dict, policy_args_dict, param_args_dict


def check_config(param_config, policy_config):
    assert not (policy_config['immediate_reward'] and policy_config['speed_reward']),\
        'Speed reward must be terminal reward'


def process_before_train(param_config=CifarConfig, policy_config=PolicyConfig):
    import pprint

    if '-h' in sys.argv or '--help' in sys.argv:
        print('Usage: add properties just like this:\n'
              '    add_label_prob=False\n'
              '    %policy_save_freq=10\n'
              '\n'
              'properties starts with % are in Config, other properties are in ParamConfig.')

    args_dict, policy_args_dict, param_args_dict = simple_parse_args(sys.argv, param_config)
    Config.update(args_dict)
    param_config.update(param_args_dict)
    policy_config.update(policy_args_dict)

    check_config(param_config, policy_config)

    message('The configures and hyperparameters are:')
    pprint.pprint(Config, stream=sys.stderr)


def test():
    pass


if __name__ == '__main__':
    test()
