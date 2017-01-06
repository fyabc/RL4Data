#! /usr/bin/python

from __future__ import print_function, unicode_literals

import traceback

from batch_updater import *
from config import CifarConfig as ParamConfig
from criticNetwork import CriticNetwork
from model_CIFAR10 import CIFARModelBase, CIFARModel, VaniliaCNNModel
from utils import *
from utils_CIFAR10 import load_cifar10_data, split_cifar10_data, iterate_minibatches, pre_process_CIFAR10_data, \
    prepare_CIFAR10_data
from policyNetwork import LRPolicyNetwork, MLPPolicyNetwork

__author__ = 'fyabc'


# TODO: Change code into updaters
# Done: raw, REINFORCE


# [NOTE]: In CIFAR10, validate point at end of each epoch.
def epoch_message(model, x_train, y_train, x_validate, y_validate, x_test, y_test,
                  history_accuracy, history_train_loss,
                  epoch, start_time, train_batches, total_accepted_cases):
    # Get training loss
    train_loss = model.get_training_loss(x_train, y_train)

    # Get validation loss and accuracy
    validate_loss, validate_acc, validate_batches = model.validate_or_test(x_validate, y_validate)
    validate_loss /= validate_batches
    validate_acc /= validate_batches
    history_accuracy.append(validate_acc)

    # Get test loss and accuracy
    test_loss, test_acc, test_batches = model.validate_or_test(x_test, y_test)
    test_loss /= test_batches
    test_acc /= test_batches

    message("Epoch {} of {} took {:.3f}s".format(epoch, ParamConfig['epoch_per_episode'], time.time() - start_time))
    message('Training Loss:', train_loss)
    message('History Training Loss:', history_train_loss / train_batches)
    message('Validate Loss:', validate_loss)
    message('#Validate accuracy:', validate_acc)
    message('Test Loss:', test_loss),
    message('#Test accuracy:', test_acc)
    message('Number of accepted cases: {} of {} total'.format(total_accepted_cases, x_train.shape[0]))


def train_raw_CIFAR10():
    model_name = eval(ParamConfig['model_name'])
    # Create neural network model
    model = model_name()
    # model = VaniliaCNNModel()

    # Load the dataset
    x_train, y_train, x_validate, y_validate, x_test, y_test, \
        train_size, validate_size, test_size = pre_process_CIFAR10_data()

    updater = RawUpdater(model, [x_train, y_train], prepare_data=prepare_CIFAR10_data)

    if ParamConfig['warm_start']:
        model.load_model(Config['model_file'])

    # Train the network
    # Some variables
    history_accuracy = []

    best_validate_acc = -np.inf
    best_iteration = 0
    test_score = 0.0
    start_time = time.time()

    for epoch in range(ParamConfig['epoch_per_episode']):
        print('[Epoch {}]'.format(epoch))
        message('[Epoch {}]'.format(epoch))

        updater.start_new_epoch()
        epoch_start_time = time.time()

        kf = get_minibatches_idx(train_size, model.train_batch_size, shuffle=True)

        for _, train_index in kf:
            part_train_cost = updater.add_batch(train_index, updater, history_accuracy)

        validate_acc, test_acc = validate_point_message(
            model, x_train, y_train, x_validate, y_validate, x_test, y_test, updater)
        history_accuracy.append(validate_acc)

        if validate_acc > best_validate_acc:
            best_validate_acc = validate_acc
            best_iteration = updater.iteration
            test_score = test_acc

        if model_name == CIFARModel:
            if (epoch + 1) in (41, 61):
                model.update_learning_rate()

        message("Epoch {} of {} took {:.3f}s".format(
            epoch, ParamConfig['epoch_per_episode'], time.time() - epoch_start_time))

    episode_final_message(best_validate_acc, best_iteration, test_score, start_time)

    if ParamConfig['save_model']:
        model.save_model()


def train_SPL_CIFAR10():
    model_name = eval(ParamConfig['model_name'])
    # Create neural network model
    model = model_name()
    # model = VaniliaCNNModel()

    # Load the dataset
    x_train, y_train, x_validate, y_validate, x_test, y_test, \
        train_size, validate_size, test_size = pre_process_CIFAR10_data()

    updater = SPLUpdater(model, [x_train, y_train], ParamConfig['epoch_per_episode'], prepare_data=prepare_CIFAR10_data)

    if ParamConfig['warm_start']:
        model.load_model(Config['model_file'])

    # Train the network
    # Some variables
    history_accuracy = []

    best_validate_acc = -np.inf
    best_iteration = 0
    test_score = 0.0
    start_time = time.time()

    for epoch in range(ParamConfig['epoch_per_episode']):
        print('[Epoch {}]'.format(epoch))
        message('[Epoch {}]'.format(epoch))

        updater.start_new_epoch()
        epoch_start_time = time.time()

        kf = get_minibatches_idx(train_size, model.train_batch_size, shuffle=True)

        for _, train_index in kf:
            part_train_cost = updater.add_batch(train_index, updater, history_accuracy)

        validate_acc, test_acc = validate_point_message(
            model, x_train, y_train, x_validate, y_validate, x_test, y_test, updater)
        history_accuracy.append(validate_acc)

        if validate_acc > best_validate_acc:
            best_validate_acc = validate_acc
            best_iteration = updater.iteration
            test_score = test_acc

        if model_name == CIFARModel:
            if (epoch + 1) in (41, 61):
                model.update_learning_rate()

        message("Epoch {} of {} took {:.3f}s".format(
            epoch, ParamConfig['epoch_per_episode'], time.time() - epoch_start_time))

    episode_final_message(best_validate_acc, best_iteration, test_score, start_time)


def train_policy_CIFAR10():
    model_name = eval(ParamConfig['model_name'])
    # Create neural network model
    model = model_name()
    # model = VaniliaCNNModel()

    # Create the policy network
    input_size = CIFARModelBase.get_policy_input_size()
    print('Input size of policy network:', input_size)
    policy_model_name = eval(PolicyConfig['policy_model_name'])
    policy = policy_model_name(input_size=input_size)
    # policy = LRPolicyNetwork(input_size=input_size)

    # Load the dataset
    x_train, y_train, x_validate, y_validate, x_test, y_test, \
        train_size, validate_size, test_size = pre_process_CIFAR10_data()

    # Train the network
    for episode in range(PolicyConfig['num_episodes']):
        print('[Episode {}]'.format(episode))
        message('[Episode {}]'.format(episode))

        policy.message_parameters()

        if ParamConfig['warm_start']:
            model.load_model(Config['model_file'])
        else:
            model.reset_parameters()
        model.reset_learning_rate()

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
            print('[Epoch {}]'.format(epoch))
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

            if model_name == CIFARModel:
                if (epoch + 1) in (41, 61):
                    model.update_learning_rate()

            message("Epoch {} of {} took {:.3f}s".format(
                epoch, ParamConfig['epoch_per_episode'], time.time() - epoch_start_time))

            # Immediate reward
            if PolicyConfig['immediate_reward']:
                validate_acc = model.get_test_acc(x_validate, y_validate)
                policy.reward_buffer.append(validate_acc)

        episode_final_message(best_validate_acc, best_iteration, test_score, start_time)

        # Updating policy
        if PolicyConfig['speed_reward']:
            terminal_reward = speed_reward_checker.get_reward()
            policy.update(terminal_reward)
        else:
            validate_acc = model.get_test_acc(x_validate, y_validate)
            policy.update(validate_acc)

        if PolicyConfig['policy_save_freq'] > 0 and episode % PolicyConfig['policy_save_freq'] == 0:
            policy.save_policy(PolicyConfig['policy_model_file'].replace('.npz', '_ep{}.npz'.format(episode)))
            policy.save_policy()


def train_actor_critic_CIFAR10():
    model_name = eval(ParamConfig['model_name'])
    # Create neural network model
    model = model_name()
    # model = VaniliaCNNModel()

    # Create the actor network
    input_size = CIFARModel.get_policy_input_size()
    print('Input size of actor network:', input_size)
    policy_model_name = eval(PolicyConfig['policy_model_name'])
    actor = policy_model_name(input_size=input_size)
    # actor = LRPolicyNetwork(input_size=input_size)
    critic = CriticNetwork(feature_size=input_size, batch_size=model.train_batch_size)

    # Load the dataset
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_cifar10_data(load_cifar10_data())
    train_small_size = ParamConfig['train_small_size']
    x_train_small, y_train_small = get_part_data(x_train, y_train, train_small_size)

    message('Training data size:', y_train_small.shape[0])
    message('Validation data size:', y_validate.shape[0])
    message('Test data size:', y_test.shape[0])

    # Train the network
    for episode in range(PolicyConfig['num_episodes']):
        print('[Episode {}]'.format(episode))
        message('[Episode {}]'.format(episode))

        actor.message_parameters()

        if ParamConfig['warm_start']:
            model.load_model(Config['model_file'])
        else:
            model.reset_parameters()
        model.reset_learning_rate()

        history_accuracy = []

        # Iteration (number of batches)
        iteration = 0

        for epoch in range(ParamConfig['epoch_per_episode']):
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

                if iteration % PolicyConfig['AC_update_freq'] == 0:
                    # Get immediate reward
                    if PolicyConfig['cost_gap_AC_reward']:
                        cost_old = part_train_err
                        cost_new = model.f_cost_without_decay(inputs, targets)
                        imm_reward = cost_old - cost_new
                    else:
                        valid_part_x, valid_part_y = get_part_data(
                            np.asarray(x_validate), np.asarray(y_validate),
                            PolicyConfig['immediate_reward_sample_size'])
                        _, valid_err, validate_batches = model.validate_or_test(valid_part_x, valid_part_y)
                        imm_reward = valid_err / validate_batches

                    # Get new state, new actions, and compute new Q value
                    probability_new = model.get_policy_input(inputs, targets, epoch, history_accuracy)
                    actions_new = actor.take_action(probability_new, log_replay=False)

                    Q_value_new = critic.Q_function(state=probability_new, action=actions_new)
                    if epoch < ParamConfig['epoch_per_episode'] - 1:
                        label = PolicyConfig['actor_gamma'] * Q_value_new + imm_reward
                    else:
                        label = imm_reward

                    # Update the critic Q network
                    Q_loss = critic.update(probability, actions, floatX(label))

                    # Update actor network
                    actor_loss = actor.update_raw(probability, actions,
                                                  np.full(actions.shape, label, dtype=probability.dtype))

                    if PolicyConfig['AC_update_freq'] >= ParamConfig['display_freq'] or \
                                            iteration % ParamConfig['display_freq'] == 0:
                        message('Epoch {}\tIteration {}\tCost {}\tCritic loss {}\tActor loss {}'
                                .format(epoch, iteration, part_train_err, Q_loss, actor_loss))

            if model_name == CIFARModel:
                if (epoch + 1) in (41, 61):
                    model.update_learning_rate()

            # add immediate reward
            if PolicyConfig['immediate_reward']:
                x_validate_small, y_validate_small = get_part_data(
                    x_validate, y_validate, PolicyConfig['immediate_reward_sample_size'])
                _, validate_acc, validate_batches = model.validate_or_test(x_validate_small, y_validate_small)
                validate_acc /= validate_batches
                actor.reward_buffer.append(validate_acc)

            epoch_message(model, x_train, y_train, x_validate, y_validate, x_test, y_test,
                          history_accuracy, history_train_loss,
                          epoch, start_time, train_batches, total_accepted_cases)

        model.test(x_test, y_test)

        validate_loss, validate_acc, validate_batches = model.validate_or_test(x_validate, y_validate)
        validate_acc /= validate_batches

        actor.update(validate_acc)

        if PolicyConfig['policy_save_freq'] > 0 and episode % PolicyConfig['policy_save_freq'] == 0:
            actor.save_policy(PolicyConfig['policy_model_file'].replace('.npz', '_ep{}.npz'.format(episode)))
            actor.save_policy()


def test_policy_CIFAR10():
    model_name = eval(ParamConfig['model_name'])
    # Create neural network model
    model = model_name()
    # model = VaniliaCNNModel()

    input_size = CIFARModel.get_policy_input_size()
    print('Input size of policy network:', input_size)

    # Load the dataset and get small training data
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_cifar10_data(load_cifar10_data())

    message('Training data size:', y_train.shape[0])
    message('Validation data size:', y_validate.shape[0])
    message('Test data size:', y_test.shape[0])

    if ParamConfig['warm_start']:
        model.load_model()

    if Config['train_type'] == 'random_drop':
        # Random drop configure
        random_drop_numbers = map(lambda l: int(l.strip()), list(open(ParamConfig['random_drop_number_file'], 'r')))
    else:
        # Build policy
        policy_model_name = eval(PolicyConfig['policy_model_name'])
        policy = policy_model_name(input_size=input_size)
        # policy = LRPolicyNetwork(input_size=input_size)
        policy.load_policy()
        policy.message_parameters()

    history_accuracy = []

    # Train the network
    for epoch in range(ParamConfig['epoch_per_episode']):
        print('[Epoch {}]'.format(epoch))
        message('[Epoch {}]'.format(epoch))

        history_train_loss = 0
        train_batches = 0
        total_accepted_cases = 0
        start_time = time.time()

        for batch in iterate_minibatches(x_train, y_train, model.train_batch_size,
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
                        1 - float(random_drop_numbers[epoch]) / len(y_train),
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

        epoch_message(model, x_train, y_train, x_validate, y_validate, x_test, y_test,
                      history_accuracy, history_train_loss,
                      epoch, start_time, train_batches, total_accepted_cases)

        model.test(x_test, y_test)


def just_ref():
    """
    This function is just refer some names to prevent them from being optimized by Pycharm.
    """

    _ = CIFARModelBase, CIFARModel, VaniliaCNNModel
    _ = LRPolicyNetwork, MLPPolicyNetwork


def main(args=None):
    process_before_train(args, ParamConfig)

    try:
        if Config['train_type'] == 'raw':
            train_raw_CIFAR10()
        elif Config['train_type'] == 'self_paced':
            train_SPL_CIFAR10()
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
    except:
        message(traceback.format_exc())
    finally:
        finalize_logging_file()


if __name__ == '__main__':
    main()
