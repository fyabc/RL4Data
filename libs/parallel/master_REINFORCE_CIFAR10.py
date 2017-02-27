#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from ..model_class.CIFAR10 import CIFARModel
from ..utility.parallel_utils import parallel_run_async
from ..utility.config import CifarConfig as ParamConfig


def main():
    parallel_run_async(CIFARModel, ParamConfig, 'episode_REINFORCE_CIFAR10.py')


if __name__ == '__main__':
    main()
