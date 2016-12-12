#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys
import os
import time
import psutil
import subprocess

from utils import process_before_train, message
from config import MNISTConfig as ParamConfig, PolicyConfig
from model_MNIST import MNISTModel
from policyNetwork import LRPolicyNetwork, MLPPolicyNetwork

__author__ = 'fyabc'


def main():
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

    args = sys.argv[i:]
    process_number = len(gpu_ids)

    process_before_train(args, ParamConfig)

    # Create the policy network
    input_size = MNISTModel.get_policy_input_size()
    message('Input size of policy network:', input_size)
    policy_model_name = eval(PolicyConfig['policy_model_name'])
    policy = policy_model_name(input_size=input_size)
    # policy = LRPolicyNetwork(input_size=input_size)

    # Save the policy before the training, because all

    episode_number = 1000

    for episode in range(episode_number):
        ret_values = [None for _ in range(process_number)]
        results = [None for _ in range(process_number)]

        envs = [os.environ.copy() for _ in range(process_number)]

        for i, env in enumerate(envs):
            env[str('THEANO_FLAGS')] = str('device=gpu{},floatX=float32'.format(gpu_ids[i]))

        pool = [
            psutil.Popen(
                ['python', 'episode_REINFORCE_MNIST.py'] + args + [
                    'logging_file=@./data/log_Pm_speed_par_ep{}_{}.npz@'.format(episode, i),
                ],
                stdout=subprocess.PIPE,
                env=envs[i],
            )
            for i in range(process_number)
        ]

        # Roll polling
        while any(e is None for e in ret_values):
            for i, process in enumerate(pool):
                ret_values[i] = process.poll()
            time.sleep(1.0)

        # Get results from standard output
        for i, process in enumerate(pool):
            if ret_values[i] == 0:
                results[i], _ = process.communicate()

        for terminal_reward in results:
            policy.update(float(terminal_reward))
        policy.save_policy()


if __name__ == '__main__':
    main()
