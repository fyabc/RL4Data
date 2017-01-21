# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import numpy as np

from utils import message
from name_register import NameRegister

__author__ = 'fyabc'

####################################
# Speed reward REINFORCE utilities #
####################################


class RewardChecker(NameRegister):
    NameTable = {}

    def check(self, validate_acc, updater):
        raise NotImplementedError()

    def get_reward(self, echo=True):
        raise NotImplementedError()


class SpeedRewardChecker(RewardChecker):
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

SpeedRewardChecker.register_class(['speed'])


class DeltaAccuracyRewardChecker(RewardChecker):
    def __init__(self, baseline_accuracy_list, weight_linear=0.0):
        self.baseline_accuracy_list = baseline_accuracy_list
        self.weight_linear = weight_linear

        self.delta_accuracy = []

    def check(self, validate_acc, updater):
        vp_number = updater.vp_number

        self.delta_accuracy.append(validate_acc - self.baseline_accuracy_list[vp_number])

    def get_reward(self, echo=True):
        # todo: get weighted reward
        pass

DeltaAccuracyRewardChecker.register_class(['delta_acc', 'delta_accuracy'])
