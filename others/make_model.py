# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys

ProjectRootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ProjectRootPath)

import numpy as np

from libs.utility.config import ModelPath
from libs.utility.utils import floatX, fX

__author__ = 'fyabc'


def make_lr_model(filename, W, b, dataset='mnist'):
    np.savez(os.path.join(ModelPath, dataset, filename), W, b)


def make_lr_force_init_model(margin, rank, bias):
    W = np.array(
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., margin, 0., rank, 0.],
        dtype=fX
    )
    b = floatX(bias)

    make_lr_model('c-mnist-{}-{}-{}-force-init.0.npz'.format(margin, rank, bias), W, b)


def main():
    make_lr_force_init_model(-3, 3, 2)


if __name__ == '__main__':
    main()

