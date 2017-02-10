#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys
import csv
import argparse

import numpy as np

__author__ = 'fyabc'


def get_line_data(line, data_list=((float, -1),)):
    words = line.split()

    result = []

    for type_, loc in data_list:
        try:
            value = type_(words[loc])
            result.append(value)
        except ValueError:
            if words[loc] == '[NotComputed]':
                result.append(None)
            else:
                raise

    if len(result) == 1:
        return result[0]
    else:
        return result


class CSVObject(object):
    Name = ''
    Headers = []
    Lines = 0
    StartTag = ''

    @classmethod
    def load(cls, line_list):
        pass

    @classmethod
    def load_list_iter(cls, f):
        f_it = iter(f)
        result = []

        try:
            while True:
                line = next(f_it)
                if line.startswith(cls.StartTag):
                    line_list = [line] + [next(f_it) for _ in range(cls.Lines - 1)]
                    result.append(cls.load(line_list))
        except StopIteration:
            pass

        return result

    @classmethod
    def dump_csv(cls, filename, data_list):
        with open(filename, 'wb') as f:
            f_csv = csv.DictWriter(f, cls.Headers)
            f_csv.writeheader()
            f_csv.writerows(data.__dict__ for data in data_list)


class ValidatePoint(CSVObject):
    """
Validate Point: Epoch 3 Iteration 10000 Batch 2500 TotalBatch 10000
Training Loss: 0.352372714296
History Training Loss: 0.365216569817
Validate Loss: 0.271493434276
#Validate accuracy: 0.924299993396
Test Loss: 0.279175536646
#Test accuracy: 0.92039999485
Number of accepted cases: 200000 of 200000 total
    """

    Name = 'VP'
    Headers = [
        'epoch', 'iteration', 'batch', 'total_batch',
        'train_loss',
        'history_train_loss',
        'validate_loss',
        'validate_accuracy',
        'test_loss',
        'test_accuracy',
        'total_cases', 'accepted_cases',
    ]
    Lines = 8
    StartTag = 'Validate Point'

    def __init__(self):
        self.epoch = None
        self.iteration = None
        self.batch = None
        self.total_batch = None

        self.train_loss = None
        self.history_train_loss = None
        self.validate_loss = None
        self.validate_accuracy = None
        self.test_loss = None
        self.test_accuracy = None
        self.accepted_cases = None
        self.total_cases = None

    def __str__(self):
        return '''\
Validate Point ?: Epoch {epoch} Iteration {iteration} Batch {batch} TotalBatch {total_batch}
Training Loss: {train_loss}
History Training Loss: {history_train_loss}
Validate Loss: {validate_loss}
#Validate accuracy: {validate_accuracy}
Test Loss: {test_loss}
#Test accuracy: {test_accuracy}
Number of accepted cases: {accepted_cases} of {total_cases} total
'''.format(**self.__dict__)

    def __repr__(self):
        return self.__str__()

    @classmethod
    def load(cls, line_list):
        result = cls.__new__(cls)

        result.epoch, result.iteration, result.batch, result.total_batch = \
            get_line_data(line_list[0], [(int, 4), (int, 6), (int, 8), (int, 10)])
        result.train_loss = get_line_data(line_list[1])
        result.history_train_loss = get_line_data(line_list[2])
        result.validate_loss = get_line_data(line_list[3])
        result.validate_accuracy = get_line_data(line_list[4])
        result.test_loss = get_line_data(line_list[5])
        result.test_accuracy = get_line_data(line_list[6])
        result.accepted_cases, result.total_cases = get_line_data(line_list[7], [(int, 4), (int, 6)])

        return result


class RewardPoint(CSVObject):
    Name = 'RP'
    Headers = [
        'first_over_cases',
        'total_cases',
        'first_over_ratio',
        'terminal_reward',
    ]
    Lines = 3
    StartTag = 'First over cases:'

    def __init__(self):
        self.first_over_cases = None
        self.total_cases = None
        self.first_over_ratio = None
        self.terminal_reward = None

    def __str__(self):
        return '''\
First over cases: {first_over_cases}
Total cases: {total_cases}
Terminal reward: {terminal_reward}
'''.format(**self.__dict__)

    @classmethod
    def load(cls, line_list):
        result = cls.__new__(cls)

        result.first_over_cases = get_line_data(line_list[0], [(int, -1)])
        result.total_cases = get_line_data(line_list[1], [(int, -1)])
        result.first_over_ratio = get_line_data(line_list[2])
        result.terminal_reward = -np.log(result.first_over_ratio)

        return result


class Reward3Point(CSVObject):
    Name = 'R3P'
    Headers = [
        'first_over_cases_95',
        'first_over_ratio_95',
        'reward_95',
        'first_over_cases_97',
        'first_over_ratio_97',
        'reward_97',
        'first_over_cases_98',
        'first_over_ratio_98',
        'reward_98',
        'total_cases',
        'terminal_reward',
    ]
    Lines = 7
    StartTag = 'Reward Point:'

    def __init__(self):
        self.first_over_cases_95 = None
        self.first_over_ratio_95 = None
        self.reward_95 = None
        self.first_over_cases_97 = None
        self.first_over_ratio_97 = None
        self.reward_97 = None
        self.first_over_cases_98 = None
        self.first_over_ratio_98 = None
        self.reward_98 = None
        self.total_cases = None
        self.terminal_reward = None

    def __str__(self):
        return '''\
Reward Point:
First over cases:
0.95 1100000 2.90042209375
0.97 2550000 2.05963891438
0.98 11700000 0.53614343175
Total cases: 20000000
Terminal reward: 2.22611453951
'''.format(**self.__dict__)

    @classmethod
    def load(cls, line_list):
        result = cls.__new__(cls)

        result.total_cases = get_line_data(line_list[5], [(int, -1)])
        result.first_over_cases_95, result.reward_95 = get_line_data(line_list[2], [(int, -2), (float, -1)])
        result.first_over_ratio_95 = float(result.first_over_cases_95) / result.total_cases
        result.first_over_cases_97, result.reward_97 = get_line_data(line_list[3], [(int, -2), (float, -1)])
        result.first_over_ratio_97 = float(result.first_over_cases_97) / result.total_cases
        result.first_over_cases_98, result.reward_98 = get_line_data(line_list[4], [(int, -2), (float, -1)])
        result.first_over_ratio_98 = float(result.first_over_cases_98) / result.total_cases
        result.terminal_reward = get_line_data(line_list[6])

        return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('raw_filename', nargs='+')
    parser.add_argument('-d', '--dataset', metavar='dataset', dest='dataset', default='m',
                        help='The name of dataset "m/c/i" (default is "m")')
    parser.add_argument('-p', '--point', metavar='point', dest='point', default='VP',
                        help='The point to extract (default is "VP (ValidatePoint)")')
    parser.add_argument('-r', '--raw', action='store_true', dest='raw', default=False,
                        help='Load the filename as real filename, not add "log_P" prefix (still add ".txt") (default is False).')

    options = parser.parse_args()

    for raw_filename in options.raw_filename:
        filename = '{}{}.txt'.format('' if options.raw else ('log_P' + options.dataset + '_'), raw_filename)

        points = {cls.Name: cls for cls in [
            ValidatePoint, RewardPoint, Reward3Point,
        ]}

        point = points[options.point]

        with open(filename, 'r') as f:
            data_list = point.load_list_iter(f)

        point.dump_csv(
            '{}_P{}_{}.csv'.format(options.point, options.dataset, raw_filename),
            data_list,
        )


if __name__ == '__main__':
    main()
