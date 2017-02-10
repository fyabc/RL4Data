#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import os
import math

__author__ = 'fyabc'

LogPath = 'D:/Others/JobResults/mnist/corrupted'


def iter_episode(filename):
    with open(os.path.join(LogPath, filename), 'r') as f:
        fi = iter(f)
        try:
            episode_acc = []
            while True:
                line = next(fi)
                if line.startswith('#Test accuracy'):
                    acc = float(line.split()[-1])
                    if acc < 0.7 and episode_acc and episode_acc[-1] > 0.9:
                        yield episode_acc
                        episode_acc = [acc]
                    else:
                        episode_acc.append(acc)
        except StopIteration:
            pass


def get_terminal_reward(episode_acc,
                        thresholds=(0.89, 0.92, 0.94),
                        weights=(1.0 / 6, 1.0 / 3, 0.5),
                        total=20000000,
                        valid_freq=5000):
    values = [None for _ in thresholds]
    for i, val in enumerate(episode_acc):
        for j, th in enumerate(thresholds):
            if values[j] is not None:
                continue
            if val >= thresholds[j]:
                values[j] = i
        if val >= thresholds[-1]:
            break

    return sum(-math.log(val * valid_freq * 1.0 / total) * weight
               for val, weight in zip(values, weights))


def main():
    episode_acc_list = list(iter_episode('log_Pm_AC_MLP_flip.txt'))

    rewards = [
        get_terminal_reward(episode_acc)
        for episode_acc in episode_acc_list
    ]

    max_reward = max(rewards)
    max_index = rewards.index(max_reward)

    print('Max AC reward:', max_reward)
    print('Max index:', max_index)


if __name__ == '__main__':
    main()
