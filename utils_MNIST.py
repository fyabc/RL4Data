#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import gzip
import numpy as np
import cPickle as pickle
import theano
import theano.tensor as T

from config import Config, MNISTConfig
from utils import fX


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

    data_dir = data_dir or MNISTConfig['data_dir']

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

    # test_set_x, test_set_y = shared_dataset(test_set)
    # valid_set_x, valid_set_y = shared_dataset(valid_set)
    # train_set_x, train_set_y = shared_dataset(train_set)
    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    rval = (train_set_x, train_set_y, valid_set_x, valid_set_y,
            test_set_x, test_set_y)
    return rval


def test():
    rval = load_mnist_data()

    for val in rval:
        print(type(val), val.shape, val.dtype)

    train_x, train_y = rval[0], rval[1]
    print(train_x[0], train_x.max(), train_x.min())
    print(train_y[:25], train_y.max(), train_y.min())


if __name__ == '__main__':
    test()
