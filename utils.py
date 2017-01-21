#! /usr/bin/python

from __future__ import print_function, unicode_literals

import cPickle as pkl
import gzip
import random
import sys
import time
import traceback
from collections import namedtuple
from functools import wraps

import numpy as np

from config import *
from path_util import get_path, split_policy_name, find_newest

__author__ = 'fyabc'


DatasetAttributes = namedtuple('DatasetAttributes', ['name', 'config', 'main_entry'])

# All datasets
Datasets = {
    'cifar10': DatasetAttributes('cifar10', CifarConfig, 'train_CIFAR10.main2'),
    'mnist': DatasetAttributes('mnist', MNISTConfig, 'train_MNIST.main2'),
    'imdb': DatasetAttributes('imdb', IMDBConfig, 'train_IMDB.main2'),
}

# The float type of Theano. Default to 'float32'.
# fX = config.floatX
fX = Config['floatX']

# Logging settings
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


# The escaped string double quote.
_StringDoubleQuote = '@'
_GlobalPrefix = 'G.'
_PolicyPrefix = 'P.'
_KeyValueSeparator = '='
_DefaultPathString = '~'


def simple_parse_args2(args):
    global_args_dict = {}
    policy_args_dict = {}
    param_args_dict = {}

    for i, arg in enumerate(args):
        arg = arg.replace(_StringDoubleQuote, '"')

        if _KeyValueSeparator in arg:
            if arg.startswith(_GlobalPrefix):
                arg = arg[2:]
                the_dict = global_args_dict
            elif arg.startswith(_PolicyPrefix):
                arg = arg[2:]
                the_dict = policy_args_dict
            else:
                the_dict = param_args_dict

            key, value = arg.split(_KeyValueSeparator)
            the_dict[key] = eval(value)
        else:
            if i > 0:
                print('Warning: The argument {} is unused'.format(arg))

    return global_args_dict, policy_args_dict, param_args_dict


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

    message('Start Time: {}'.format(time.ctime()))
    message('The configures and hyperparameters are:')
    pprint.pprint(Config, stream=sys.stderr)
    if logging_file != sys.stderr:
        pprint.pprint(Config, stream=logging_file)


def _strict_update(target, new_dict):
    for k, v in new_dict.iteritems():
        if k not in target:
            raise KeyError('The key {} is not in the parameters.'.format(k))
        target[k] = v


def process_before_train2(args=None):
    """

    Parameters
    ----------
    args

    Returns
    -------
    A DatasetAttributes instance, indicates the dataset information.
    """

    args = args or sys.argv

    import pprint
    import platform

    if '-h' in args or '--help' in args:
        # TODO add more help message
        exit()

    global_args_dict, policy_args_dict, param_args_dict = simple_parse_args2(args)

    _strict_update(Config, global_args_dict)
    _strict_update(PolicyConfig, policy_args_dict)

    dataset_attr = Datasets[Config['dataset'].lower()]
    ParamConfig = dataset_attr.config

    _strict_update(ParamConfig, param_args_dict)

    check_config(ParamConfig, PolicyConfig)

    # # Get basename of files.
    # basename_p_save = os.path.basename(PolicyConfig['policy_save_file'])
    # basename_p_load = os.path.basename(PolicyConfig['policy_load_file'])
    # basename_log = os.path.basename(Config['logging_file'])

    # Replace _DefaultPathString('~') with real path.
    model_path = get_path(ModelPath, dataset_attr.name)
    log_path = get_path(LogPath, dataset_attr.name)
    PolicyConfig['policy_save_file'] = PolicyConfig['policy_save_file'].replace(_DefaultPathString, model_path)
    PolicyConfig['policy_load_file'] = PolicyConfig['policy_load_file'].replace(_DefaultPathString, model_path)
    Config['logging_file'] = Config['logging_file'].replace(_DefaultPathString, log_path)

    # [NOTE] The train action.
    train_action = Config['action'].lower()
    train_type = Config['train_type'].lower()

    if train_type in NoPolicyTypes:
        # Job without policy, do nothing.
        append = False

    elif train_type in CommonTypes:
        # Common mode:
        #     Used for training without policy (raw/SPL/test/random_drop)
        #     Options:
        #         Create a new logging file
        #         Load a exist model (if needed) from {P.policy_load_file}
        #         If the episode is specified (e.g. {P.policy_load_file = '~/model.14.npz'}), just load it
        #         else (e.g. {P.policy_load_file = '~/model.npz'}), load the newest model.
        raw_name, episode, ext = split_policy_name(PolicyConfig['policy_load_file'])
        if episode == '':
            # Load the newest model
            PolicyConfig['policy_load_file'] = find_newest(model_path, raw_name, ext)
        append = False

    elif train_action == 'overwrite':
        # Overwrite mode:
        #     Used for starting a new training policy, overwrite old models if exists.
        #     Options:
        #         Creating a new logging file
        PolicyConfig['start_episode'] = -1
        append = False

    elif train_action == 'reload':
        # Reload mode:
        #     Used for reload a job.
        #     Options:
        #         Append to an exist logging file
        #         Load model setting is like common mode.
        raw_name, episode, ext = split_policy_name(PolicyConfig['policy_load_file'])

        if episode == '':
            # Load the newest model
            PolicyConfig['policy_load_file'], PolicyConfig['start_episode'] = find_newest(
                model_path, raw_name, ext, ret_number=True)
        append = True

    else:
        raise KeyError('Unknown train action {}'.format(train_action))

    init_logging_file(append=append)

    # Set random seed.
    np.random.seed(Config['seed'])

    message('[Message before train]')
    message('Running on node: {}'.format(platform.node()))
    message('Start Time: {}'.format(time.ctime()))

    message('The configures and hyperparameters are:')
    pprint.pprint(Config, stream=sys.stderr)
    if logging_file != sys.stderr:
        pprint.pprint(Config, stream=logging_file)

    message('[Message before train done]')

    return dataset_attr


def call_or_throw(call_dict, key, *args, **kwargs):     # Unused now
    func = call_dict.get(key, None)

    if func is None:
        raise KeyError('Unknown entry name {}'.format(key))

    return func(*args, **kwargs)


def dataset_main(call_table):
    try:
        train_func = call_table.get(Config['train_type'].lower(), None)

        if train_func is None:
            raise KeyError('Unknown train type {}'.format(Config['train_type']))

        train_func()
    except:
        message(traceback.format_exc())
    finally:
        process_after_train()


def process_after_train():
    message('[Message after train]')
    message('End Time: {}'.format(time.ctime()))
    message('[Message after train done]')
    finalize_logging_file()


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


def validate_point_message(
        model,
        x_train, y_train, x_validate, y_validate, x_test, y_test,
        updater,
        reward_checker=None,
):
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
Validate Point {}: Epoch {} Iteration {} Batch {} TotalBatch {}
Training Loss: {}
History Training Loss: {}
Validate Loss: {}
#Validate accuracy: {}
Test Loss: {}
#Test accuracy: {}
Number of accepted cases: {} of {} total""".format(
        updater.vp_number, updater.epoch, updater.iteration, updater.epoch_train_batches, updater.total_train_batches,
        train_loss,
        updater.epoch_history_train_loss / updater.epoch_train_batches,
        validate_loss,
        validate_acc,
        test_loss,
        test_acc,
        updater.total_accepted_cases, updater.total_seen_cases,
    ))

    # Check speed rewards
    if reward_checker is not None:
        reward_checker.check(validate_acc, updater)

    # [NOTE] Important! increment `vp_number` in validation point.
    # `DeltaAccuracyRewardChecker` need `vp_number` to work correctly.
    updater.vp_number += 1

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


def _test_initialize():
    process_before_train2()


def _test():
    # _test_logging_file()
    _test_initialize()


if __name__ == '__main__':
    _test()
