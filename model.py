#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

__author__ = 'fyabc'


class ModelBase(object):
    def build_train_function(self):
        raise NotImplementedError()

    def build_validate_function(self):
        raise NotImplementedError()

    def validate_or_test(self, x_test, y_test):
        raise NotImplementedError()
