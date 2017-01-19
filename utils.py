#! /usr/bin/python

from __future__ import print_function, unicode_literals

import cPickle as pkl
import os
import random
import sys
import time
from functools import wraps
import gzip

import numpy as np

from config import Config, CifarConfig, PolicyConfig

__author__ = 'fyabc'

# fX = config.floatX
fX = Config['floatX']

logging_file = sys.stderr

_depth = 0


def init_logging_file(append=False):
    global logging_file

    if Config['logging_file'] is None:
        return

    if append:
        logging_file = open(Config['logging_file'], 'a')
        return

    raw_filename = Config['logging_file']
    i = 1

    filename = raw_filename

    while os.path.exists(filename):
        filename = raw_filename.replace('.txt', '_{}.txt'.format(i))
        i += 1

    Config['logging_file'] = filename
    logging_file = open(filename, 'w')


def finalize_logging_file():
    if logging_file != sys.stderr:
        logging_file.flush()
        logging_file.close()


def message(*args, **kwargs):
    if logging_file != sys.stderr:
        print(*args, file=logging_file, **kwargs)
    print(*args, file=sys.stderr, **kwargs)


def logging(func, file_=sys.stderr):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _depth

        message(' ' * 2 * _depth + '[Start function %s...]' % func.__name__)
        _depth += 1
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        _depth -= 1
        message(' ' * 2 * _depth + '[Function %s done, time: %.3fs]' % (func.__name__, end_time - start_time))
        return result
    return wrapper


def floatX(value):
    return np.asarray(value, dtype=fX)


def init_norm(*dims, **kwargs):
    normalize = kwargs.pop('normalize', True)
    result = floatX(np.random.randn(*dims))
    if normalize:
        result /= np.sqrt(result.size)
    return result


def unpickle(filename):
    if filename.endswith('.gz'):
        _open = gzip.open
    else:
        _open = open

    with _open(filename, 'rb') as f:
        return pkl.load(f)


def average(sequence):
    if sequence is None:
        return 0.0
    if len(sequence) == 0:
        return 0.0
    return sum(sequence) / len(sequence)


def get_rank(a):
    temp = a.argsort()
    ranks = np.empty_like(a)
    ranks[temp] = np.arange(len(a))

    return ranks


###############################
# Data loading and processing #
###############################

def get_part_data(x_data, y_data, part_size=None):
    if part_size is None:
        return x_data, y_data

    train_size = x_data.shape[0]
    if train_size < part_size:
        return x_data, y_data

    # Use small dataset to check the code
    sampled_indices = random.sample(range(train_size), part_size)
    return x_data[sampled_indices], y_data[sampled_indices]


def shuffle_data(x_train, y_train):
    shuffled_indices = np.arange(y_train.shape[0])
    np.random.shuffle(shuffled_indices)
    return x_train[shuffled_indices], y_train[shuffled_indices]


####################################
# Speed reward REINFORCE utilities #
####################################

class SpeedRewardChecker(object):
    def __init__(self, check_point_list, expected_total_cases):
        """

        Parameters
        ----------
        check_point_list : list of 2-element tuples
            format: [(threshold1, weight1), (threshold2, weight2), ...]
        """

        self._check_point_list = check_point_list
        self.thresholds = [check_point[0] for check_point in check_point_list]
        self.weights = [check_point[1] for check_point in check_point_list]
        self.first_over_cases = [None for _ in range(len(check_point_list))]
        self.expected_total_cases = expected_total_cases

    @property
    def num_checker(self):
        return len(self.thresholds)

    def check(self, validate_acc, updater):
        for i, threshold in enumerate(self.thresholds):
            if self.first_over_cases[i] is None and validate_acc >= threshold:
                self.first_over_cases[i] = updater.total_accepted_cases

    def get_reward(self, echo=True):
        for i in range(len(self.first_over_cases)):
            if self.first_over_cases[i] is None:
                self.first_over_cases[i] = self.expected_total_cases

        terminal_rewards = [-np.log(float(first_over_cases) / self.expected_total_cases)
                            for first_over_cases in self.first_over_cases]
        result = sum(weight * terminal_reward for weight, terminal_reward in zip(self.weights, terminal_rewards))

        if echo:
            message('Reward Point:\nFirst over cases:')
            for threshold, first_over_cases, terminal_reward in zip(
                    self.thresholds, self.first_over_cases, terminal_rewards):
                message('{} {} {}'.format(threshold, first_over_cases, terminal_reward))
            message('Total cases:', self.expected_total_cases)
            message('Terminal reward:', result)

        return result


class DeltaAccuracyRewardChecker(object):
    def __init__(self, baseline_accuracy_list, weight_linear=0.0):
        self.baseline_accuracy_list = baseline_accuracy_list


########################################
# Simple command line arguments parser #
########################################

def simple_parse_args(args, param_config=CifarConfig):
    args_dict = {}
    policy_args_dict = {}
    param_args_dict = {}

    for arg in args:
        arg = arg.replace('@', '"')

        if '=' in arg:
            if arg.startswith('G.'):
                arg = arg[2:]
                the_dict = args_dict
                target_dict = Config
            elif arg.startswith('P.'):
                arg = arg[2:]
                the_dict = policy_args_dict
                target_dict = PolicyConfig
            else:
                the_dict = param_args_dict
                target_dict = param_config
            key, value = arg.split('=')
            if key not in target_dict:
                raise Exception('The key {} is not in the parameters.'.format(key))

            the_dict[key] = eval(value)

    return args_dict, policy_args_dict, param_args_dict


def check_config(param_config, policy_config):
    assert not (policy_config['immediate_reward'] and policy_config['speed_reward']),\
        'Speed reward must be terminal reward'


def process_before_train(args=None, param_config=CifarConfig, policy_config=PolicyConfig, append=None):
    args = args or sys.argv

    import pprint

    if '-h' in args or '--help' in args:
        # TODO add more help message
        exit()

    args_dict, policy_args_dict, param_args_dict = simple_parse_args(args, param_config)
    Config.update(args_dict)
    param_config.update(param_args_dict)
    policy_config.update(policy_args_dict)

    check_config(param_config, policy_config)

    if append is None:
        append = Config['append_logging_file']

    init_logging_file(append=append)

    message('The configures and hyperparameters are:')
    pprint.pprint(Config, stream=sys.stderr)
    if logging_file != sys.stderr:
        pprint.pprint(Config, stream=logging_file)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != n:
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return list(enumerate(minibatches))


def get_policy(model_type, policy_type, save=True):
    """Create the policy network"""

    input_size = model_type.get_policy_input_size()
    message('Input size of policy network:', input_size)

    policy = policy_type(input_size=input_size)

    if save:
        policy.save_policy()

    return policy


def validate_point_message(model, x_train, y_train, x_validate, y_validate, x_test, y_test, updater):
    # Get training loss
    train_loss = model.get_training_loss(x_train, y_train)

    # Get validation loss and accuracy
    validate_loss, validate_acc, validate_batches = model.validate_or_test(x_validate, y_validate)
    validate_loss /= validate_batches
    validate_acc /= validate_batches

    # Get test loss and accuracy
    # [NOTE]: In this version, test at each validate point is fixed.
    test_loss, test_acc, test_batches = model.validate_or_test(x_test, y_test)
    test_loss /= test_batches
    test_acc /= test_batches

    message("""\
Validate Point: Epoch {} Iteration {} Batch {} TotalBatch {}
Training Loss: {}
History Training Loss: {}
Validate Loss: {}
#Validate accuracy: {}
Test Loss: {}
#Test accuracy: {}
Number of accepted cases: {} of {} total""".format(
        updater.epoch, updater.iteration, updater.epoch_train_batches, updater.total_train_batches,
        train_loss,
        updater.epoch_history_train_loss / updater.epoch_train_batches,
        validate_loss,
        validate_acc,
        test_loss,
        test_acc,
        updater.total_accepted_cases, updater.total_seen_cases,
    ))

    if Config['temp_job'] == 'check_selected_data_label':
        message("""\
Epoch label count: {}
Total label count: {}""".format(
            updater.epoch_label_count,
            updater.total_label_count,
        ))
    elif Config['temp_job'] == 'log_data':
        message('LogData', end='\t')
        message(*updater.updated_indices, sep='\t')

    return validate_acc, test_acc


def episode_final_message(best_validate_acc, best_iteration, test_score, start_time):
    message('$Final results:')
    message('$  best test accuracy:\t\t{} %'.format((test_score * 100.0) if test_score is not None else None))
    message('$  best validation accuracy: {}'.format(best_validate_acc))
    message('$  obtained at iteration {}'.format(best_iteration))
    message('$  Time passed: {:.2f}s'.format(time.time() - start_time))


def _test_logging_file():
    global logging_file

    logging_file = open('./data/temp.txt', 'w')

    message('Test logging')


def _test():
    _test_logging_file()


if __name__ == '__main__':
    _test()
