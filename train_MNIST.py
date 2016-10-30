#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import traceback

from utils import process_before_train, message, finalize_logging_file
from config import Config, MNISTConfig


def train_raw_MNIST():
    pass


def train_SPL_MNIST():
    pass


def train_policy_MNIST():
    pass


def train_actor_critic_MNIST():
    pass


def test_policy_MNIST():
    pass


if __name__ == '__main__':
    process_before_train(MNISTConfig)

    try:
        if Config['train_type'] == 'raw':
            train_raw_MNIST()
        elif Config['train_type'] == 'self_paced':
            train_SPL_MNIST()
        elif Config['train_type'] == 'policy':
            train_policy_MNIST()
        elif Config['train_type'] == 'actor_critic':
            train_actor_critic_MNIST()
        elif Config['train_type'] == 'deterministic':
            test_policy_MNIST()
        elif Config['train_type'] == 'stochastic':
            test_policy_MNIST()
        elif Config['train_type'] == 'random_drop':
            test_policy_MNIST()
        else:
            raise Exception('Unknown train type {}'.format(Config['train_type']))
    except:
        message(traceback.format_exc())
    finally:
        finalize_logging_file()
