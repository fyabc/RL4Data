#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from functools import partial

from ..utility.utils import *
from ..model_class.NMT import NMTModel
from ..utility.NMT import TextIterator
from ..utility.config import NMTConfig as ParamConfig, Config, PolicyConfig

__author__ = 'fyabc'


def train_raw_NMT_template(train_type='raw'):
    text_iterator = TextIterator(
        ParamConfig['data_src'],
        ParamConfig['data_tgt'],
        ParamConfig['vocab_src_filename'],
        ParamConfig['vocab_tgt_filename'],
        ParamConfig['batch_size'],
        ParamConfig['maxlen'],
        ParamConfig['n_words_src'],
        ParamConfig['n_words'],
    )

    model = NMTModel()

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
