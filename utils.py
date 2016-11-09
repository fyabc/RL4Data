#! /usr/bin/python

from __future__ import print_function, unicode_literals

import cPickle as pkl
import os
import random
import sys
import time
from functools import wraps

import numpy as np

from config import Config, CifarConfig, PolicyConfig

__author__ = 'fyabc'

# fX = config.floatX
fX = Config['floatX']

logging_file = sys.stderr

_depth = 0


def init_logging_file():
    global logging_file

    if Config['logging_file'] is None:
        return

    raw_filename = Config['logging_file']
    i = 1

    filename = raw_filename

    while os.path.exists(filename):
        filename = raw_filename.replace('.txt', '{}.txt'.format(i))
        i += 1

    Config['logging_file'] = filename
    logging_file = open(filename, 'w')


def finalize_logging_file():
    if logging_file != sys.stderr:
        logging_file.flush()
        logging_file.close()


def message(*args, **kwargs):
    if logging_file != sys.stderr:
        print(*args, file=logging_file, **kwargs)
    print(*args, file=sys.stderr, **kwargs)


def logging(func, file_=sys.stderr):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _depth

        message(' ' * 2 * _depth + '[Start function %s...]' % func.__name__)
        _depth += 1
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        _depth -= 1
        message(' ' * 2 * _depth + '[Function %s done, time: %.3fs]' % (func.__name__, end_time - start_time))
        return result
    return wrapper


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


def get_rank(a):
    temp = a.argsort()
    ranks = np.empty_like(a)
    ranks[temp] = np.arange(len(a))

    return ranks


###############################
# Data loading and processing #
###############################

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
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
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


def process_before_train(args=None, param_config=CifarConfig, policy_config=PolicyConfig):
    args = args or sys.argv

    import pprint

    if '-h' in args or '--help' in args:
        # TODO add more help message
        exit()

    args_dict, policy_args_dict, param_args_dict = simple_parse_args(args, param_config)
    Config.update(args_dict)
    param_config.update(param_args_dict)
    policy_config.update(policy_args_dict)

    check_config(param_config, policy_config)

    init_logging_file()

    message('The configures and hyperparameters are:')
    pprint.pprint(Config, stream=sys.stderr)
    if logging_file != sys.stderr:
        pprint.pprint(Config, stream=logging_file)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != n:
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return list(enumerate(minibatches))


def test():
    global logging_file

    logging_file = open('./data/temp.txt', 'w')

    message('Test logging')


if __name__ == '__main__':
    test()