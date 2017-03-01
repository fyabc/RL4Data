#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from functools import partial

from ..utility.utils import *

__author__ = 'fyabc'


def train_raw_NMT_template(train_type='raw'):
    pass

train_raw_NMT = partial(train_raw_NMT_template, 'raw')
train_SPL_NMT = partial(train_raw_NMT_template, 'spl')
test_stochastic_NMT = partial(train_raw_NMT_template, 'stochastic')
test_deterministic_NMT = partial(train_raw_NMT_template, 'deterministic')
test_random_drop_NMT = partial(train_raw_NMT_template, 'random_drop')


def main():
    dataset_main({
        'raw': None,
        'self_paced': None,
        'spl': None,

        'policy': None,
        'reinforce': None,
        'speed': None,

        'actor_critic': None,
        'ac': None,

        # 'test': None,
        'deterministic': None,
        'stochastic': None,
        'random_drop': None,

        'new_train': None,
    })
