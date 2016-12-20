#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from config import CifarConfig as ParamConfig
from model_CIFAR10 import CIFARModel
from parallel_utils import parallel_run_async


def main():
    parallel_run_async(CIFARModel, ParamConfig, 'episode_REINFORCE_CIFAR10.py')


if __name__ == '__main__':
    main()
