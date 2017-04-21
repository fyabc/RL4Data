#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import os
from itertools import count

from utils import load_list, save_list

__author__ = 'fyabc'


# TODO: Change this hard code!
LogPath = 'D:/Others/JobResults/mnist/corrupted'
DataPath = './data/mnist_flip'


def main(**kwargs):
    filename = kwargs.pop('filename', 'raw_flip')
    target_name = kwargs.pop('target', 'raw')
    one_file = kwargs.pop('one_file', True)

    acc = []

    if one_file:
        file_list = [os.path.join(LogPath, 'log_Pm_{}.txt'.format(filename))]
    else:
        file_list = []
        for i in count(1):
            real_filename = os.path.join(LogPath, 'log_Pm_{}_{}.txt'.format(filename, i))
            if not os.path.exists(real_filename):
                break
            file_list.append(real_filename)

    for real_filename in file_list:
        with open(real_filename, 'r') as f:
            for idx, line in enumerate(line for line in f if line.startswith('#Test accuracy')):
                val = float(line.split()[-1])
                if len(acc) <= idx:
                    acc.append([1, val])
                else:
                    e = acc[idx]
                    e[0] += 1
                    e[1] += val

    acc = [e[1] / e[0] for e in acc]

    save_list(acc, os.path.join(DataPath, '{}_tests_accuracy.txt'.format(target_name)))
    save_list(range(5000, 5000 * len(acc) + 1, 5000),
              os.path.join(DataPath, '{}_case_numbers.txt'.format(target_name)))


if __name__ == '__main__':
    # main(filename='raw_flip', target='raw')
    # main(filename='SPL_flip_5000vp', target='SPL')
    main(filename='random_drop_speed_flip', target='random_drop_speed')
