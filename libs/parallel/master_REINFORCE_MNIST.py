#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from ..model_class.MNIST import MNISTModel
from ..utility.parallel_utils import parallel_run_async
from ..utility.config import MNISTConfig as ParamConfig

__author__ = 'fyabc'


def main():
    parallel_run_async(MNISTModel, ParamConfig, 'episode_REINFORCE_MNIST.py')


if __name__ == '__main__':
    main()
