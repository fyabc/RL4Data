#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import traceback

from config import MNISTConfig as ParamConfig, PolicyConfig
from utils import *
from utils_MNIST import pre_process_MNIST_data
from model_MNIST import MNISTModel
from policyNetwork import LRPolicyNetwork, MLPPolicyNetwork
from batch_updater import TrainPolicyUpdater

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
    # validation_frequency = min(train_size // model.train_batch_size, patience // 2)
    validation_frequency = ParamConfig['valid_freq']

    return patience, patience_increase, improvement_threshold, validation_frequency


def train_episode():
    model = MNISTModel()

    # Create the policy network
    input_size = MNISTModel.get_policy_input_size()
    message('Input size of policy network:', input_size)
    policy_model_name = eval(PolicyConfig['policy_model_name'])
    policy = policy_model_name(input_size=input_size)
    # policy = LRPolicyNetwork(input_size=input_size)

    # At the start of this episode, load the policy.
    policy.load_policy()

    # Load the dataset and config
    x_train, y_train, x_validate, y_validate, x_test, y_test, \
        train_size, validate_size, test_size = pre_process_MNIST_data()
    patience, patience_increase, improvement_threshold, validation_frequency = pre_process_config(model, train_size)

    # Train the network
    # Some variables
    history_accuracy = []

    # To prevent the double validate point
    last_validate_point = -1

    # get small training data
    x_train_small, y_train_small = get_part_data(x_train, y_train, ParamConfig['train_small_size'])
    train_small_size = len(x_train_small)
    message('Training small size:', train_small_size)

    # Speed reward
    speed_reward_checker = SpeedRewardChecker(
        PolicyConfig['speed_reward_config'],
        ParamConfig['epoch_per_episode'] * train_small_size,
    )

    updater = TrainPolicyUpdater(model, [x_train_small, y_train_small], policy)

    best_validate_acc = -np.inf
    best_iteration = 0
    test_score = 0.0
    start_time = time.time()

    for epoch in range(ParamConfig['epoch_per_episode']):
        message('[Epoch {}]'.format(epoch))

        updater.start_new_epoch()
        epoch_start_time = time.time()

        kf = get_minibatches_idx(train_small_size, model.train_batch_size, shuffle=True)

        for _, train_index in kf:
            part_train_cost = updater.add_batch(train_index, updater, history_accuracy)

            if updater.total_train_batches > 0 and \
                    updater.total_train_batches != last_validate_point and \
                    updater.total_train_batches % validation_frequency == 0:
                last_validate_point = updater.total_train_batches
                validate_acc, test_acc = validate_point_message(
                    model, x_train, y_train, x_validate, y_validate, x_test, y_test, updater)
                history_accuracy.append(validate_acc)

                if validate_acc > best_validate_acc:
                    # improve patience if loss improvement is good enough
                    if (1. - validate_acc) < (1. - best_validate_acc) * improvement_threshold:
                        patience = max(patience, updater.iteration * patience_increase)
                    best_validate_acc = validate_acc
                    best_iteration = updater.iteration
                    test_score = test_acc

                # Check speed rewards
                speed_reward_checker.check(validate_acc, updater)

            if updater.total_train_batches >= patience:
                break

        message("Epoch {} of {} took {:.3f}s".format(
            epoch, ParamConfig['epoch_per_episode'], time.time() - epoch_start_time))

        if updater.iteration >= patience:
            break

    episode_final_message(best_validate_acc, best_iteration, test_score, start_time)

    # Print terminal reward
    terminal_reward = speed_reward_checker.get_reward()
    print(terminal_reward)


def just_ref():
    """
    This function is just refer some names to prevent them from being optimized by Pycharm.
    """

    _ = LRPolicyNetwork, MLPPolicyNetwork


def main(args=None):
    process_before_train(args, ParamConfig)

    try:
        train_episode()
    except:
        message(traceback.format_exc())
    finally:
        finalize_logging_file()


if __name__ == '__main__':
    main()
