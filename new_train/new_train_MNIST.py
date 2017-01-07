# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys
import time

import numpy as np

from local_utils import get_project_root_path

sys.path.append(get_project_root_path())

from config import MNISTConfig as ParamConfig, PolicyConfig
from utils import process_before_train, message, episode_final_message
from model_MNIST import MNISTModel
from utils_MNIST import pre_process_MNIST_data, pre_process_config

__author__ = 'fyabc'


def new_train_MNIST():
    model = MNISTModel()

    # Create the policy network
    input_size = MNISTModel.get_policy_input_size()
    # todo

    # Load the dataset and config
    x_train, y_train, x_validate, y_validate, x_test, y_test, train_size, validate_size, test_size = pre_process_MNIST_data()
    patience, patience_increase, improvement_threshold, validation_frequency = pre_process_config(model, train_size)

    for episode in range(PolicyConfig['num_episodes']):
        print('[Episode {}]'.format(episode))
        message('[Episode {}]'.format(episode))

        model.reset_parameters()

        # Train the network
        # Some variables
        history_accuracy = []

        # To prevent the double validate point
        last_validate_point = -1

        # todo

        best_validate_acc = -np.inf
        best_iteration = 0
        test_score = 0.0
        start_time = time.time()

        for epoch in range(ParamConfig['epoch_per_episode']):
            pass

        episode_final_message(best_validate_acc, best_iteration, test_score, start_time)
