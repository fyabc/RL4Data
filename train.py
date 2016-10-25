#! /usr/bin/python

from __future__ import print_function, unicode_literals

import sys
import heapq

import numpy as np

from config import Config, CifarConfig, PolicyConfig
from utils import *
from model_CIFAR10 import CIFARModel, VaniliaCNNModel
from policyNetwork import PolicyNetwork
from criticNetwork import CriticNetwork

__author__ = 'fyabc'


def train_raw_CIFAR10():
    model_name = eval(CifarConfig['model_name'])
    # Create neural network model
    model = model_name()
    # model = VaniliaCNNModel()

    # Load the dataset
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_cifar10_data(load_cifar10_data())

    message('Training data size:', y_train.shape[0])
    message('Validation data size:', y_validate.shape[0])
    message('Test data size:', y_test.shape[0])

    if Config['train_type'] == 'self_paced':
        # Self-paced learning iterate on data cases
        total_iteration_number = CifarConfig['epoch_per_episode'] * len(x_train) // model.train_batch_size

        # Get the cost threshold \lambda.
        def cost_threshold(iteration):
            return 1 + (model.train_batch_size - 1) * iteration / total_iteration_number

    # Train the network
    if CifarConfig['warm_start']:
        model.load_model(Config['model_file'])
    else:
        model.reset_parameters()

    # Iteration (number of batches)
    iteration = 0

    for epoch in range(CifarConfig['epoch_per_episode']):
        print('[Epoch {}]'.format(epoch))
        message('[Epoch {}]'.format(epoch))

        total_accepted_cases = 0
        history_train_loss = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(x_train, y_train, CifarConfig['train_batch_size'], shuffle=True, augment=True):
            iteration += 1

            inputs, targets = batch

            if Config['train_type'] == 'self_paced':
                selected_number = cost_threshold(iteration)

                cost_list = model.f_cost_list_without_decay(inputs, targets)
                label_cost_lists = [cost_list[targets == i] for i in range(10)]

                actions = np.full(targets.shape, False, dtype=bool)

                for i, label_cost_list in enumerate(label_cost_lists):
                    if label_cost_list.size != 0:
                        threshold = heapq.nsmallest(selected_number, label_cost_list)[-1]
                        for j in range(len(targets)):
                            if targets[j] == i and cost_list[j] <= threshold:
                                actions[j] = True

                # Get masked inputs and targets
                inputs = inputs[actions]
                targets = targets[actions]

            total_accepted_cases += len(inputs)

            part_train_err = model.f_train(inputs, targets)

            if iteration % CifarConfig['display_freq'] == 0:
                message('Train error of iteration {} is {}'.format(iteration, part_train_err))

            history_train_loss += part_train_err
            train_batches += 1

        validate_loss, validate_acc, validate_batches = model.validate_or_test(x_validate, y_validate)
        validate_acc /= validate_batches

        # Get training loss
        train_loss = model.get_training_loss(x_train, y_train)

        message("Epoch {} of {} took {:.3f}s".format(epoch, CifarConfig['epoch_per_episode'], time.time() - start_time))
        message('Training Loss:', train_loss)
        message('History Training Loss:', history_train_loss / train_batches)
        message('Validate Loss:', validate_loss / validate_batches)
        message('#Validate accuracy:', validate_acc)

        message('Number of accepted cases {} of {} cases'.format(total_accepted_cases, len(x_train)))

        if model_name == CIFARModel:
            if (epoch + 1) in (41, 61):
                model.update_learning_rate()

    if Config['save_model'] and validate_acc >= 0.35:
        message('Saving CNN model warm start... ', end='')
        model.save_model()
        message('done')

    model.test(x_test, y_test)


def train_policy_CIFAR10():
    model_name = eval(CifarConfig['model_name'])
    # Create neural network model
    model = model_name()
    # model = VaniliaCNNModel()

    # Create the policy network
    input_size = CIFARModel.get_policy_input_size()
    print('Input size of policy network:', input_size)
    policy = PolicyNetwork(input_size=input_size)

    # Load the dataset
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_cifar10_data(load_cifar10_data())
    train_small_size = CifarConfig['train_small_size']
    x_train_small, y_train_small = get_part_data(x_train, y_train, train_small_size)

    message('Training data size:', y_train_small.shape[0])
    message('Validation data size:', y_validate.shape[0])
    message('Test data size:', y_test.shape[0])

    # Train the network
    for episode in range(1, PolicyConfig['num_episodes'] + 1):
        print('[Episode {}]'.format(episode))
        message('[Episode {}]'.format(episode))

        if CifarConfig['warm_start']:
            model.load_model(Config['model_file'])
        else:
            model.reset_parameters()
        model.reset_learning_rate()

        history_accuracy = []

        for epoch in range(CifarConfig['epoch_per_episode']):
            print('[Epoch {}]'.format(epoch))
            message('[Epoch {}]'.format(epoch))

            policy.start_new_epoch()

            total_accepted_cases = 0
            history_train_loss = 0
            train_batches = 0
            start_time = time.time()

            for batch in iterate_minibatches(x_train_small, y_train_small, model.train_batch_size,
                                             shuffle=True, augment=True):
                inputs, targets = batch

                probability = model.get_policy_input(inputs, targets, epoch, history_accuracy)
                actions = policy.take_action(probability)

                # get masked inputs and targets
                inputs = inputs[actions]
                targets = targets[actions]

                total_accepted_cases += len(inputs)

                history_train_loss += model.f_train(inputs, targets)
                train_batches += 1

            validate_loss, validate_acc, validate_batches = model.validate_or_test(x_validate, y_validate)
            validate_acc /= validate_batches

            history_accuracy.append(validate_acc)

            if model_name == CIFARModel:
                if (epoch + 1) in (41, 61):
                    model.update_learning_rate()

            # add immediate reward
            if PolicyConfig['immediate_reward']:
                x_validate_small, y_validate_small = get_part_data(
                    x_validate, y_validate, PolicyConfig['immediate_reward_sample_size'])
                _, validate_acc, validate_batches = model.validate_or_test(x_validate_small, y_validate_small)
                validate_acc /= validate_batches
                policy.reward_buffer.append(validate_acc)

            # Get training loss
            train_loss = model.get_training_loss(x_train, y_train)

            message("Epoch {} of {} took {:.3f}s"
                    .format(epoch, CifarConfig['epoch_per_episode'], time.time() - start_time))
            message('Training Loss:', train_loss)
            message('History Training Loss:', history_train_loss / train_batches)
            message('Validate Loss:', validate_loss / validate_batches)
            message('#Validate accuracy:', validate_acc)

            message('Number of accepted cases {} of {} cases'.format(total_accepted_cases, train_small_size))

        model.test(x_test, y_test)

        validate_loss, validate_acc, validate_batches = model.validate_or_test(x_validate, y_validate)
        validate_acc /= validate_batches

        policy.update(validate_acc)

        if Config['policy_save_freq'] > 0 and episode % Config['policy_save_freq'] == 0:
            policy.save_policy()

        if episode % PolicyConfig['policy_learning_rate_discount_freq'] == 0:
            policy.discount_learning_rate()


def train_actor_critic_CIFAR10():
    model_name = eval(CifarConfig['model_name'])
    # Create neural network model
    model = model_name()
    # model = VaniliaCNNModel()

    # Create the actor network
    input_size = CIFARModel.get_policy_input_size()
    print('Input size of actor network:', input_size)
    actor = PolicyNetwork(input_size=input_size)
    critic = CriticNetwork(feature_size=input_size, batch_size=model.train_batch_size)

    # Load the dataset
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_cifar10_data(load_cifar10_data())
    train_small_size = CifarConfig['train_small_size']
    x_train_small, y_train_small = get_part_data(x_train, y_train, train_small_size)

    message('Training data size:', y_train_small.shape[0])
    message('Validation data size:', y_validate.shape[0])
    message('Test data size:', y_test.shape[0])

    # Train the network
    for episode in range(1, PolicyConfig['num_episodes'] + 1):
        print('[Episode {}]'.format(episode))
        message('[Episode {}]'.format(episode))

        if CifarConfig['warm_start']:
            model.load_model(Config['model_file'])
        else:
            model.reset_parameters()
        model.reset_learning_rate()

        history_accuracy = []

        # Iteration (number of batches)
        iteration = 0

        for epoch in range(CifarConfig['epoch_per_episode']):
            print('[Epoch {}]'.format(epoch))
            message('[Epoch {}]'.format(epoch))

            actor.start_new_epoch()

            total_accepted_cases = 0
            history_train_loss = 0
            train_batches = 0
            start_time = time.time()

            for batch in iterate_minibatches(x_train_small, y_train_small, model.train_batch_size,
                                             shuffle=True, augment=True):
                iteration += 1

                inputs, targets = batch

                probability = model.get_policy_input(inputs, targets, epoch, history_accuracy)
                actions = actor.take_action(probability)

                # get masked inputs and targets
                inputs_selected = inputs[actions]
                targets_selected = targets[actions]

                total_accepted_cases += len(inputs_selected)

                # Update the CIFAR10 network with selected data
                part_train_err = model.f_train(inputs_selected, targets_selected)
                history_train_loss += part_train_err
                train_batches += 1

                # Get immediate reward
                if PolicyConfig['cost_gap_AC_reward']:
                    cost_old = part_train_err
                    cost_new = model.f_cost_without_decay(inputs, targets)
                    imm_reward = cost_old - cost_new
                else:
                    valid_part_x, valid_part_y = get_part_data(
                        np.asarray(x_validate), np.asarray(y_validate), PolicyConfig['immediate_reward_sample_size'])
                    _, valid_err, validate_batches = model.validate_or_test(valid_part_x, valid_part_y)
                    imm_reward = valid_err / validate_batches

                # Get new state, new actions, and compute new Q value
                probability_new = model.get_policy_input(inputs, targets, epoch, history_accuracy)
                actions_new = actor.take_action(probability_new, log_replay=False)

                Q_value_new = critic.Q_function(state=probability_new, action=actions_new)
                if epoch < CifarConfig['epoch_per_episode'] - 1:
                    label = PolicyConfig['actor_gamma'] * Q_value_new + imm_reward
                else:
                    label = imm_reward

                # Update the critic Q network
                Q_loss = critic.update(probability, actions, floatX(label))

                # Update actor network
                actor_loss = actor.update_raw(probability, actions,
                                              np.full(actions.shape, label, dtype=probability.dtype))

                if iteration % CifarConfig['display_freq'] == 0:
                    print('Epoch {}\tIteration {}\tCost {}\tCritic loss {}\tActor loss {}'
                          .format(epoch, iteration, part_train_err, Q_loss, actor_loss))

            validate_loss, validate_acc, validate_batches = model.validate_or_test(x_validate, y_validate)
            validate_acc /= validate_batches

            history_accuracy.append(validate_acc)

            if model_name == CIFARModel:
                if (epoch + 1) in (41, 61):
                    model.update_learning_rate()

            # Get training loss
            train_loss = model.get_training_loss(x_train, y_train)

            # add immediate reward
            if PolicyConfig['immediate_reward']:
                x_validate_small, y_validate_small = get_part_data(
                    x_validate, y_validate, PolicyConfig['immediate_reward_sample_size'])
                _, validate_acc, validate_batches = model.validate_or_test(x_validate_small, y_validate_small)
                validate_acc /= validate_batches
                actor.reward_buffer.append(validate_acc)

            print("Epoch {} of {} took {:.3f}s"
                  .format(epoch, CifarConfig['epoch_per_episode'], time.time() - start_time))
            print('Training Loss:', train_loss)
            print('History Training Loss:', history_train_loss / train_batches)
            print('Validate Loss:', validate_loss / validate_batches)
            print('#Validate accuracy:', validate_acc)

            message('Number of accepted cases {} of {} cases'.format(total_accepted_cases, train_small_size))

        model.test(x_test, y_test)

        validate_loss, validate_acc, validate_batches = model.validate_or_test(x_validate, y_validate)
        validate_acc /= validate_batches

        actor.update(validate_acc)

        if Config['policy_save_freq'] > 0 and episode % Config['policy_save_freq'] == 0:
            actor.save_policy()

        if episode % PolicyConfig['policy_learning_rate_discount_freq'] == 0:
            actor.discount_learning_rate()


def test_policy_CIFAR10():
    model_name = eval(CifarConfig['model_name'])
    # Create neural network model
    model = model_name()
    # model = VaniliaCNNModel()

    input_size = CIFARModel.get_policy_input_size()
    print('Input size of policy network:', input_size)

    # Load the dataset and get small training data
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_cifar10_data(load_cifar10_data())
    x_train_small, y_train_small = get_part_data(x_train, y_train)

    message('Training data size:', y_train_small.shape[0])
    message('Validation data size:', y_validate.shape[0])
    message('Test data size:', y_test.shape[0])

    if CifarConfig['warm_start']:
        model.load_model()

    if Config['train_type'] == 'random_drop':
        # Random drop configure
        random_drop_numbers = map(lambda l: int(l.strip()), list(open(CifarConfig['random_drop_number_file'], 'r')))
    else:
        # Build policy
        policy = PolicyNetwork(input_size=input_size)
        policy.load_policy()
        message('$    w = {}\n'
                '$    b = {}'
                .format(policy.W.get_value(), policy.b.get_value()))

    history_accuracy = []

    # Train the network
    for epoch in range(CifarConfig['epoch_per_episode']):
        print('[Epoch {}]'.format(epoch))
        message('[Epoch {}]'.format(epoch))

        history_train_loss = 0
        train_batches = 0
        total_accepted_cases = 0
        start_time = time.time()

        for batch in iterate_minibatches(x_train_small, y_train_small, model.train_batch_size,
                                         shuffle=True, augment=True):
            inputs, targets = batch

            probability = model.get_policy_input(inputs, targets, epoch, history_accuracy)

            if Config['train_type'] == 'deterministic':
                alpha = np.asarray([policy.output_function(prob) for prob in probability], dtype=fX)

                alpha /= np.sum(alpha)

                history_train_loss += model.f_alpha_train(inputs, targets, alpha)
            else:
                if Config['train_type'] == 'stochastic':
                    actions = policy.take_action(probability, False)
                elif Config['train_type'] == 'random_drop':
                    actions = np.random.binomial(
                        1,
                        1 - float(random_drop_numbers[epoch]) / CifarConfig['train_small_size'],
                        targets.shape
                    ).astype(bool)

                # get masked inputs and targets
                inputs = inputs[actions]
                targets = targets[actions]

                history_train_loss += model.f_train(inputs, targets)

            total_accepted_cases += len(inputs)
            train_batches += 1

        if model_name == CIFARModel:
            if (epoch + 1) in (41, 61):
                model.update_learning_rate()

        # Get training loss
        train_loss = model.get_training_loss(x_train, y_train)

        validate_loss, validate_acc, validate_batches = model.validate_or_test(x_validate, y_validate)
        validate_acc /= validate_batches
        history_accuracy.append(validate_acc)

        print("Epoch {} of {} took {:.3f}s".format(epoch, CifarConfig['epoch_per_episode'], time.time() - start_time))
        print('Training Loss:', train_loss)
        print('History Training Loss:', history_train_loss / train_batches)
        print('Validate Loss:', validate_loss / validate_batches)
        print('#Validate accuracy:', validate_acc)
        print('Number of accepted cases: {} of {} total'.format(total_accepted_cases, x_train_small.shape[0]))

        model.test(x_test, y_test)


if __name__ == '__main__':
    process_before_train(CifarConfig)

    try:
        if Config['train_type'] == 'raw':
            train_raw_CIFAR10()
        elif Config['train_type'] == 'self_paced':
            train_raw_CIFAR10()
        elif Config['train_type'] == 'policy':
            train_policy_CIFAR10()
        elif Config['train_type'] == 'actor_critic':
            train_actor_critic_CIFAR10()
        elif Config['train_type'] == 'deterministic':
            test_policy_CIFAR10()
        elif Config['train_type'] == 'stochastic':
            test_policy_CIFAR10()
        elif Config['train_type'] == 'random_drop':
            test_policy_CIFAR10()
        else:
            raise Exception('Unknown train type {}'.format(Config['train_type']))
    except Exception as e:
        message()
        finalize_logging_file()
        raise
