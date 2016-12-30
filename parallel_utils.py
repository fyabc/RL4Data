#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys
import os
import cPickle as pkl
import time
import gc
import subprocess
import psutil
import numpy as np

from utils import process_before_train, message, get_policy
from config import PolicyConfig, Config
from policyNetwork import LRPolicyNetwork, MLPPolicyNetwork


def get_gpu_id():
    gpu_ids = []

    argc = len(sys.argv)

    i = 1
    for i in range(1, argc):
        arg = sys.argv[i]
        try:
            gpu_id = int(arg)
            gpu_ids.append(gpu_id)
        except ValueError:
            break

    remain_args = sys.argv[i:]

    return gpu_ids, remain_args


def parallel_run_async(model_type, param_config, slave_script_name):
    gpu_ids, args = get_gpu_id()
    process_number = len(gpu_ids)

    process_before_train(args, param_config)

    policy = get_policy(model_type, eval(PolicyConfig['policy_model_name']), save=True)

    if param_config['warm_start'] is True:
        policy.load_policy(param_config['warm_start_model_file'])

    episode_number = 1000
    for episode in range(episode_number):
        ret_values = [None for _ in range(process_number)]
        results = [None for _ in range(process_number)]

        envs = [os.environ.copy() for _ in range(process_number)]

        for i, env in enumerate(envs):
            env[str('THEANO_FLAGS')] = str('device=gpu{},floatX=float32'.format(gpu_ids[i]))

        pool = []

        # Add new child process after sleep some time to avoid from compile lock.
        for i in range(process_number):
            pool.append(
                psutil.Popen(
                    ['python', slave_script_name] + args + [
                        'G.logging_file=@{}@'
                        .format(Config['logging_file'])
                        .replace('.txt', '_p{}.txt'.format(i)),
                    ],
                    stdout=subprocess.PIPE,
                    env=envs[i],
                )
            )

            time.sleep(100)

        # Roll polling
        while any(e is None for e in ret_values):
            time.sleep(1.0)

            for i, process in enumerate(pool):
                if ret_values[i] is not None:
                    continue
                ret_values[i] = process.poll()

                # Both None and non-zero
                if ret_values[i] != 0:
                    continue

                if ret_values[i] == 0:
                    results[i], _ = process.communicate()

                message('Loading and removing temp file... ', end='')
                with open(results[i], 'rb') as f:
                    terminal_reward, input_buffer, action_buffer = pkl.load(f)
                os.remove(results[i])
                message('done')

                gc.collect()

                policy.input_buffer = input_buffer
                policy.action_buffer = action_buffer
                policy.update(terminal_reward)

        policy.save_policy(PolicyConfig['policy_model_file'].replace('.npz', '_ep{}.npz'.format(episode)))
        policy.save_policy()


def parallel_run_sync(model_type, param_config, slave_script_name):
    gpu_ids, args = get_gpu_id()
    process_number = len(gpu_ids)

    process_before_train(args, param_config)

    policy = get_policy(model_type, eval(PolicyConfig['policy_model_name']), save=True)

    episode_number = 1000
    for episode in range(episode_number):
        ret_values = [None for _ in range(process_number)]
        results = [None for _ in range(process_number)]

        envs = [os.environ.copy() for _ in range(process_number)]

        for i, env in enumerate(envs):
            env[str('THEANO_FLAGS')] = str('device=gpu{},floatX=float32'.format(gpu_ids[i]))

        pool = []

        # Add new child process after sleep some time to avoid from compile lock.
        for i in range(process_number):
            pool.append(
                psutil.Popen(
                    ['python', slave_script_name] + args + [
                        'G.logging_file=@{}@'
                    .format(Config['logging_file'])
                    .replace('.txt', '_p{}.txt'.format(i)),
                    ],
                    stdout=subprocess.PIPE,
                    env=envs[i],
                )
            )

            time.sleep(100)

        # Roll polling
        while any(e is None for e in ret_values):
            time.sleep(1.0)

            for i, process in enumerate(pool):
                ret_values[i] = process.poll()

        # Model average
        avg_parameters = [
            np.zeros_like(parameter.get_value())
            for parameter in policy.parameters
            ]

        for i, process in enumerate(pool):
            results[i], _ = process.communicate()

            message('Loading and removing temp policy file... ', end='')
            with np.load(results[i]) as f:
                for j in range(len(avg_parameters)):
                    avg_parameters[j] += f['arr_{}'.format(j)]
            os.remove(results[i])
            message('done')

        for i in range(len(avg_parameters)):
            avg_parameters[i] /= process_number
            policy.parameters[i].set_value(avg_parameters[i])

        policy.save_policy(PolicyConfig['policy_model_file'].replace('.npz', '_ep{}.npz'.format(episode)))
        policy.save_policy()
