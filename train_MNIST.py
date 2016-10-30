#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys
import heapq
import traceback
from collections import deque

import numpy as np

from config import Config, MNISTConfig as ParamConfig, PolicyConfig
from utils import *
from utils_MNIST import load_mnist_data
from model_MNIST import MNISTModel
from policyNetwork import PolicyNetwork
from criticNetwork import CriticNetwork

__author__ = 'fyabc'


def pre_process_config(model, train_size):
    # Some hyperparameters
    # early-stopping parameters
    # look as this many examples regardless
    patience = ParamConfig['patience']
    # wait this much longer when a new best is found
    patience_increase = ParamConfig['patience_increase']
    # a relative improvement of this much is considered significant
    improvement_threshold = ParamConfig['improvement_threshold']

    # go through this many minibatches before checking the network
    # on the validation set; in this case we check every epoch
    validation_frequency = min(train_size // model.train_batch_size, patience // 2)
    # validation_frequency = 1

    return patience, patience_increase, improvement_threshold, validation_frequency


def validate_point_message(model, x_train, y_train, x_validate, y_validate, x_test, y_test,
                           history_train_loss, train_batches, total_accepted_cases, epoch, iteration):
    # Get training loss
    train_loss = model.get_training_loss(x_train, y_train)

    # Get validation loss and accuracy
    validate_loss, validate_acc, validate_batches = model.validate_or_test(x_validate, y_validate)
    validate_loss /= validate_batches
    validate_acc /= validate_batches

    # Get test loss and accuracy
    test_loss, test_acc, test_batches = model.validate_or_test(x_test, y_test)
    test_loss /= test_batches
    test_acc /= test_batches

    message('Validate Point: Epoch {} Iteration {}'.format(epoch, iteration))
    message('Training Loss:', train_loss)
    message('History Training Loss:', history_train_loss / train_batches)
    message('Validate Loss:', validate_loss)
    message('#Validate accuracy:', validate_acc)
    message('Test Loss:', test_loss),
    message('#Test accuracy:', test_acc)
    message('Number of accepted cases: {} of {} total'.format(
        total_accepted_cases, train_batches * model.train_batch_size))

    return validate_acc, test_acc


def episode_final_message(best_validation_acc, best_iteration, test_score, start_time):
    message('$Final results:')
    message('$  best test accuracy:\t\t{:.2f} %'.format(test_score * 100.0))
    message('$  best validation accuracy: {}'.format(best_validation_acc))
    message('$  obtained at iteration {}'.format(best_iteration))
    message('$  Time passed: {:.2f}s'.format(time.time() - start_time))


def train_raw_MNIST():
    model = MNISTModel()

    # Load the dataset
    x_train, y_train, x_validate, y_validate, x_test, y_test = load_mnist_data()

    train_size, validate_size, test_size = y_train.shape[0], y_validate.shape[0], y_test.shape[0]

    message('Training data size:', train_size)
    message('Validation data size:', validate_size)
    message('Test data size:', test_size)

    patience, patience_increase, improvement_threshold, validation_frequency = pre_process_config(model, train_size)

    # Train the network
    # Some variables
    # Iteration (number of batches)
    iteration = 0

    total_accepted_cases = 0

    best_validation_acc = -np.inf
    best_iteration = 0
    test_score = 0.
    start_time = time.time()

    for epoch in range(ParamConfig['epoch_per_episode']):
        print('[Epoch {}]'.format(epoch))
        message('[Epoch {}]'.format(epoch))

        total_accepted_cases = 0
        history_train_loss = 0
        train_batches = 0
        epoch_start_time = time.time()

        kf = get_minibatches_idx(train_size, model.train_batch_size, shuffle=True)

        for _, train_index in kf:
            iteration += 1

            inputs, targets = x_train[train_index], y_train[train_index]

            total_accepted_cases += len(inputs)

            part_train_cost = model.f_train(inputs, targets)
            history_train_loss += part_train_cost

            train_batches += 1

            if iteration % validation_frequency == 0:
                validate_acc, test_acc = validate_point_message(
                    model, x_train, y_train, x_validate, y_validate, x_test, y_test,
                    history_train_loss, train_batches, total_accepted_cases, epoch, iteration)
                # if we got the best validation score until now
                if validate_acc > best_validation_acc:
                    # improve patience if loss improvement is good enough
                    if (1. - validate_acc) < (1. - best_validation_acc) * improvement_threshold:
                        patience = max(patience, iteration * patience_increase)
                    best_validation_acc = validate_acc
                    best_iteration = iteration

                    test_score = test_acc

            if iteration >= patience:
                break

        message("Epoch {} of {} took {:.3f}s".format(
            epoch, ParamConfig['epoch_per_episode'], time.time() - epoch_start_time))
        if iteration >= patience:
            break

    episode_final_message(best_validation_acc, best_iteration, test_score, start_time)


def train_SPL_MNIST():
    model = MNISTModel()

    # Load the dataset
    x_train, y_train, x_validate, y_validate, x_test, y_test = load_mnist_data()

    train_size, validate_size, test_size = y_train.shape[0], y_validate.shape[0], y_test.shape[0]

    message('Training data size:', train_size)
    message('Validation data size:', validate_size)
    message('Test data size:', test_size)

    patience, patience_increase, improvement_threshold, validation_frequency = pre_process_config(model, train_size)

    # Train the network
    # Some variables
    # Iteration (number of batches)
    iteration = 0

    total_accepted_cases = 0

    best_validation_acc = -np.inf
    best_iteration = 0
    test_score = 0.
    start_time = time.time()

    for epoch in range(ParamConfig['epoch_per_episode']):
        print('[Epoch {}]'.format(epoch))
        message('[Epoch {}]'.format(epoch))

        total_accepted_cases = 0
        history_train_loss = 0
        train_batches = 0
        epoch_start_time = time.time()

        kf = get_minibatches_idx(train_size, model.train_batch_size, shuffle=True)

        for _, train_index in kf:
            iteration += 1

            inputs, targets = x_train[train_index], y_train[train_index]

            total_accepted_cases += len(inputs)

            part_train_cost = model.f_train(inputs, targets)
            history_train_loss += part_train_cost

            train_batches += 1

            if iteration % validation_frequency == 0:
                validate_acc, test_acc = validate_point_message(
                    model, x_train, y_train, x_validate, y_validate, x_test, y_test,
                    history_train_loss, train_batches, total_accepted_cases, epoch, iteration)
                # if we got the best validation score until now
                if validate_acc > best_validation_acc:
                    # improve patience if loss improvement is good enough
                    if (1. - validate_acc) < (1. - best_validation_acc) * improvement_threshold:
                        patience = max(patience, iteration * patience_increase)
                    best_validation_acc = validate_acc
                    best_iteration = iteration

                    test_score = test_acc

            if iteration >= patience:
                break

        message("Epoch {} of {} took {:.3f}s".format(
            epoch, ParamConfig['epoch_per_episode'], time.time() - epoch_start_time))
        if iteration >= patience:
            break

    episode_final_message(best_validation_acc, best_iteration, test_score, start_time)


def train_policy_MNIST():
    pass


def train_actor_critic_MNIST():
    pass


def test_policy_MNIST():
    pass


if __name__ == '__main__':
    process_before_train(ParamConfig)

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
