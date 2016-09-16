# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

from IMDB import IMDBModel
from utils import *

__author__ = 'fyabc'


def train_IMDB():
    # Loading data
    train_data, valid_data, test_data = load_imdb_data()
    train_data, valid_data, test_data = preprocess_imdb_data(train_data, valid_data, test_data)

    train_x, train_y = train_data
    valid_x, valid_y = valid_data
    test_x, test_y = test_data

    print("%d train examples" % len(train_x))
    print("%d valid examples" % len(valid_x))
    print("%d test examples" % len(test_x))

    # Building model
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray

    # init params and t-params

    imdb = IMDBModel(IMDBConfig['reload_model'])

    # Training
    imdb.train(train_x, train_y, valid_x, valid_y, test_x, test_y)


if __name__ == '__main__':
    process_before_train(IMDBConfig)

    train_IMDB()
