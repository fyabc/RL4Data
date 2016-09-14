#! /usr/bin/python

from __future__ import print_function, unicode_literals

from config import IMDBConfig

__author__ = 'fyabc'


class IMDBModel(object):
    def __init__(self):
        self.train_batch_size = IMDBConfig['train_batch_size']
        self.validate_batch_size = IMDBConfig['validate_batch_size']
