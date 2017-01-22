#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from batch_updater import TrainPolicyUpdater
from config import CifarConfig as ParamConfig
from model_CIFAR10 import CIFARModel
from policy_network import LRPolicyNetwork, MLPPolicyNetwork
from reward_checker import SpeedRewardChecker
from utils import *
from utils_CIFAR10 import pre_process_CIFAR10_data, prepare_CIFAR10_data


def train_episode():
    model = CIFARModel()

    policy = get_policy(CIFARModel, eval(PolicyConfig['policy_model_type']), save=False)

    # At the start of this episode, load the policy.
    policy.load_policy()

    # Load the dataset
    x_train, y_train, x_validate, y_validate, x_test, y_test, \
        train_size, validate_size, test_size = pre_process_CIFAR10_data()

    # Train the network
    # Some variables
    history_accuracy = []

    # get small training data
    x_train_small, y_train_small = get_part_data(x_train, y_train, ParamConfig['train_small_size'])
    train_small_size = len(x_train_small)
    message('Training small size:', train_small_size)

    # Speed reward
    speed_reward_checker = SpeedRewardChecker(
        PolicyConfig['speed_reward_config'],
        ParamConfig['epoch_per_episode'] * train_small_size,
    )

    updater = TrainPolicyUpdater(model, [x_train_small, y_train_small], policy, prepare_data=prepare_CIFAR10_data)

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

        validate_acc, test_acc = validate_point_message(
            model, x_train, y_train, x_validate, y_validate, x_test, y_test, updater)
        history_accuracy.append(validate_acc)

        if validate_acc > best_validate_acc:
            best_validate_acc = validate_acc
            best_iteration = updater.iteration
            test_score = test_acc

        # Check speed rewards
        speed_reward_checker.check(validate_acc, updater)

        if True:
            if (epoch + 1) in (41, 61):
                model.update_learning_rate()

        message("Epoch {} of {} took {:.3f}s".format(
            epoch, ParamConfig['epoch_per_episode'], time.time() - epoch_start_time))

    episode_final_message(best_validate_acc, best_iteration, test_score, start_time)

    terminal_reward = speed_reward_checker.get_reward()

    random_filename = './data/temp_Pc_speed_par_{}.npz'.format(np.random.randint(0, 10000000))
    with open(random_filename, 'wb') as f:
        pkl.dump((terminal_reward, policy.input_buffer, policy.action_buffer), f)

    print(random_filename, end='')


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
