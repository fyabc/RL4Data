#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from config import MNISTConfig as ParamConfig
from model_MNIST import MNISTModel
from parallel_utils import parallel_run_async

__author__ = 'fyabc'


def main():
    parallel_run_async(MNISTModel, ParamConfig, 'episode_REINFORCE_MNIST.py')


if __name__ == '__main__':
    main()
