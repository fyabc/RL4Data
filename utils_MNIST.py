#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import gzip
import numpy as np
import cPickle as pickle
import theano
import theano.tensor as T

from config import Config, MNISTConfig as ParamConfig
from utils import fX, get_part_data
from logging_utils import message


def load_mnist_data(data_dir=None):
    """ Loads the dataset

    :type data_dir: string
    :param data_dir: the path to the dataset (here MNIST)

    return data:
        x_train: (50000, 784), float32, 0.0 ~ 1.0
        y_train: (50000,), int64, 0 ~ 9
        x_validate: (10000, 784), ...
        y_validate: (10000,), ...
        x_test: (10000, 784), ...
        y_test: (10000,), ...
    """

    data_dir = data_dir or ParamConfig['data_dir']

    # Load the dataset
    with gzip.open(data_dir, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a np.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # np.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=fX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=fX), borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    # x_test, y_test = shared_dataset(test_set)
    # x_valid, y_valid = shared_dataset(valid_set)
    # x_train, y_train = shared_dataset(train_set)
    x_test, y_test = test_set
    x_valid, y_valid = valid_set
    x_train, y_train = train_set

    if Config['filter_data'] == 'random_80':
        # Randomly drop 20% data before train
        train_size = len(y_train)
        x_train, y_train = get_part_data(x_train, y_train, train_size * 8 // 10)
    elif Config['filter_data'] == 'worst_80':
        # Drop best 20% data before train
        train_size = len(y_train)
        x_train = x_train[train_size * 2 // 10:]
        y_train = y_train[train_size * 2 // 10:]
    elif Config['filter_data'] == 'best_80':
        # Drop worst 20% data before train
        train_size = len(y_train)
        x_train = x_train[:-train_size * 2 // 10]
        y_train = y_train[:-train_size * 2 // 10]

    rval = (x_train, y_train, x_valid, y_valid,
            x_test, y_test)
    return rval


def pre_process_MNIST_data():
    # Load the dataset
    x_train, y_train, x_validate, y_validate, x_test, y_test = load_mnist_data()

    train_size, validate_size, test_size = y_train.shape[0], y_validate.shape[0], y_test.shape[0]

    message('Training data size:', train_size)
    message('Validation data size:', validate_size)
    message('Test data size:', test_size)

    return x_train, y_train, x_validate, y_validate, x_test, y_test, train_size, validate_size, test_size


def test():
    rval = load_mnist_data()

    for val in rval:
        print(type(val), val.shape, val.dtype)

    train_x, train_y = rval[0], rval[1]
    print(train_x[0], train_x.max(), train_x.min())
    print(train_y[:25], train_y.max(), train_y.min())


if __name__ == '__main__':
    test()


def pre_process_config(model, train_size):
    # Some hyperparameters
    # early-stopping parameters
    # look as this many examples regardless
    patience = ParamConfig['patience']
    # wait this much longer when a new best is found
    patience_increase = ParamConfig['patience_increase']
    # a relative improvement of this much is considered significant
    improvement_threshold = ParamConfig['improvement_threshold']

    # go through this many minibatches before checking the network
    # on the validation set; in this case we check every epoch
    # validation_frequency = min(train_size // model.train_batch_size, patience // 2)
    validation_frequency = ParamConfig['valid_freq']

    return patience, patience_increase, improvement_threshold, validation_frequency