#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from collections import deque
import heapq

import numpy as np


class BatchUpdater(object):
    def __init__(self, model, all_data):
        self.batch_size = model.train_batch_size

        # Data buffer.
        self.buffer = deque()

        # The model.
        self.model = model

        # Data. (a tuple, elements are x, y, ...)
        self.all_data = all_data

        # Iteration (number of batches)
        self.iteration = 0
        self.epoch = -1

        self.epoch_train_batches = 0
        self.epoch_accepted_cases = 0
        self.epoch_history_train_loss = 0.0

        self.total_train_batches = 0
        self.total_accepted_cases = 0

    @property
    def data_size(self):
        return len(self.all_data[0])

    @property
    def total_seen_cases(self):
        return self.batch_size * self.iteration

    def filter_batch(self, batch_index, *args):
        """get the filtered indices in the batch.

        Parameters
        ----------
        batch_index: list of int
            a batch of indices in all_data.
        args: a list of some args
            other arguments required by policy or SPL.
            args[0]: int
                epoch
            args[1]: list of float
                history_accuracy

        Returns
        -------
        list of int
            a list of selected indices.
        """
        pass

    def start_new_epoch(self):
        self.epoch += 1
        self.epoch_train_batches = 0
        self.epoch_accepted_cases = 0
        self.epoch_history_train_loss = 0.0

    def train_batch_buffer(self):
        update_batch_index = [self.buffer.popleft() for _ in range(self.batch_size)]

        selected_batch_data = [data[update_batch_index] for data in self.all_data]

        part_train_cost = self.model.f_train(*selected_batch_data)

        self.epoch_accepted_cases += len(selected_batch_data[0])
        self.epoch_train_batches += 1

        self.total_accepted_cases += len(selected_batch_data[0])
        self.total_train_batches += 1

        self.epoch_history_train_loss += part_train_cost

        return part_train_cost

    def add_batch(self, batch_index, *args):
        self.iteration += 1

        selected_index = self.filter_batch(batch_index, *args)

        self.buffer.extend(selected_index)

        if len(self.buffer) >= self.batch_size:
            return self.train_batch_buffer()
        else:
            return None


class RawUpdater(BatchUpdater):
    def __init__(self, model, all_data):
        super(RawUpdater, self).__init__(model, all_data)

    def filter_batch(self, batch_index, *args):
        return list(batch_index)


class SPLUpdater(BatchUpdater):
    def __init__(self, model, all_data, epoch_per_episode):
        """

        Parameters
        ----------
        model
        all_data : list
            [NOTE]: The target must be the last one of the data list.
        epoch_per_episode
        """

        super(SPLUpdater, self).__init__(model, all_data)

        self.expected_total_iteration = epoch_per_episode * self.data_size // self.model.train_batch_size

    def cost_threshold(self, iteration):
        return 1 + (self.model.train_batch_size - 1) * iteration // self.expected_total_iteration

    def filter_batch(self, batch_index, *args):
        selected_number = self.cost_threshold(self.iteration)
        selected_batch_data = [data[batch_index] for data in self.all_data]

        targets = selected_batch_data[-1]

        cost_list = self.model.f_cost_list_without_decay(*selected_batch_data)
        label_cost_lists = [cost_list[targets == label] for label in range(self.model.output_size)]

        result = []

        for i, label_cost_list in enumerate(label_cost_lists):
            if label_cost_list.size != 0:
                threshold = heapq.nsmallest(selected_number, label_cost_list)[-1]
                for j in range(len(targets)):
                    if targets[j] == i and cost_list[j] <= threshold:
                        result.append(batch_index[j])

        return result


class TrainPolicyUpdater(BatchUpdater):
    def __init__(self, model, all_data, policy):
        super(TrainPolicyUpdater, self).__init__(model, all_data)
        self.policy = policy

    def start_new_epoch(self):
        super(TrainPolicyUpdater, self).start_new_epoch()
        self.policy.start_new_epoch()

    def filter_batch(self, batch_index, *args):
        selected_batch_data = [data[batch_index] for data in self.all_data]
        selected_batch_data.extend(args)

        probability = self.model.get_policy_input(*selected_batch_data)
        action = self.policy.take_action(probability)

        return [index for i, index in enumerate(batch_index) if action[i]]


class TestPolicyUpdater(BatchUpdater):
    def __init__(self, model, all_data, policy):
        super(TestPolicyUpdater, self).__init__(model, all_data)
        self.policy = policy

    def filter_batch(self, batch_index, *args):
        selected_batch_data = [data[batch_index] for data in self.all_data]
        selected_batch_data.extend(args)

        probability = self.model.get_policy_input(*selected_batch_data)
        action = self.policy.take_action(probability, False)

        return [index for i, index in enumerate(batch_index) if action[i]]


class RandomDropUpdater(BatchUpdater):
    def __init__(self, model, all_data, random_drop_number_file):
        super(RandomDropUpdater, self).__init__(model, all_data)
        self.random_drop_numbers = map(lambda l: int(l.strip()), list(open(random_drop_number_file, 'r')))

    def filter_batch(self, batch_index, *args):
        epoch = args[0]
        action = np.random.binomial(
            1,
            1 - float(self.random_drop_numbers[epoch]) / self.data_size,
            self.all_data[-1].shape
        ).astype(bool)

        return [index for i, index in enumerate(batch_index) if action[i]]
