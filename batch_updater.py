#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from collections import deque
import heapq

import numpy as np

from extensions import PartLossChecker
from config import Config
from utils import message


class BatchUpdater(object):
    def __init__(self, model, all_data, **kwargs):
        """

        Parameters
        ----------
        model
        all_data
        kwargs :
            prepare_data:
        """

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

        if Config['temp_job'] == 'check_selected_data_label':
            self.epoch_label_count = np.zeros((self.model.output_size,), dtype='int64')
            self.total_label_count = np.zeros((self.model.output_size,), dtype='int64')
        elif Config['temp_job'] == 'check_part_loss':
            self.part_loss_checker = PartLossChecker(self)

        # A hook: the last update batch index.
        self.last_update_batch_index = None

        # The prepare data hook
        self.prepare_data = kwargs.get('prepare_data', lambda *data: data)

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

        if Config['temp_job'] == 'check_selected_data_label':
            self.epoch_label_count.fill(0)

    def train_batch_buffer(self):
        self.last_update_batch_index = [self.buffer.popleft() for _ in range(self.batch_size)]

        selected_batch_data = [data[self.last_update_batch_index] for data in self.all_data]
        selected_batch_data = self.prepare_data(*selected_batch_data)

        if Config['temp_job'] == 'check_selected_data_label':
            selected_batch_label = selected_batch_data[-1]
            for i in range(len(self.epoch_label_count)):
                count_i = sum(selected_batch_label == i)
                self.epoch_label_count[i] += count_i
                self.total_label_count[i] += count_i

        part_train_cost = self.model.f_train(*selected_batch_data)

        self.epoch_accepted_cases += len(selected_batch_data[0])
        self.epoch_train_batches += 1

        self.total_accepted_cases += len(selected_batch_data[0])
        self.total_train_batches += 1

        if Config['temp_job'] == 'check_part_loss':
            self.part_loss_checker.check()

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
    def __init__(self, model, all_data, **kwargs):
        super(RawUpdater, self).__init__(model, all_data, **kwargs)

    def filter_batch(self, batch_index, *args):
        return list(batch_index)


class SPLUpdater(BatchUpdater):
    def __init__(self, model, all_data, epoch_per_episode, **kwargs):
        """

        Parameters
        ----------
        model
        all_data : list
            [NOTE]: The target must be the last one of the data list.
        epoch_per_episode
        """

        super(SPLUpdater, self).__init__(model, all_data, **kwargs)

        self.expected_total_iteration = epoch_per_episode * self.data_size // self.model.train_batch_size

        if Config['temp_job'] == 'log_data':
            self.updated_indices = [0 for _ in range(10)]

    def cost_threshold(self, iteration):
        return 1 + (self.model.train_batch_size - 1) * iteration // self.expected_total_iteration

    def filter_batch(self, batch_index, *args):
        selected_number = self.cost_threshold(self.iteration)

        selected_batch_data = [data[batch_index] for data in self.all_data]
        selected_batch_data = self.prepare_data(*selected_batch_data)

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
                        if Config['temp_job'] == 'log_data':
                            self.updated_indices[batch_index[j] // 5000] += 1

        if Config['temp_job'] == 'log_data':
            message(*self.updated_indices, sep='\t')

        return result


class TrainPolicyUpdater(BatchUpdater):
    def __init__(self, model, all_data, policy, **kwargs):
        super(TrainPolicyUpdater, self).__init__(model, all_data, **kwargs)
        self.policy = policy

    def start_new_epoch(self):
        super(TrainPolicyUpdater, self).start_new_epoch()
        self.policy.start_new_epoch()

    def filter_batch(self, batch_index, *args):
        selected_batch_data = [data[batch_index] for data in self.all_data]
        selected_batch_data = self.prepare_data(*selected_batch_data)

        probability = self.model.get_policy_input(*(selected_batch_data + args))
        action = self.policy.take_action(probability, True)

        result = [index for i, index in enumerate(batch_index) if action[i]]

        return result


class ACUpdater(BatchUpdater):
    def __init__(self, model, all_data, policy, **kwargs):
        super(ACUpdater, self).__init__(model, all_data, **kwargs)
        self.policy = policy

        self.last_probability = None
        self.last_action = None

    def start_new_epoch(self):
        super(ACUpdater, self).start_new_epoch()
        self.policy.start_new_epoch()

    def filter_batch(self, batch_index, *args):
        selected_batch_data = [data[batch_index] for data in self.all_data]
        selected_batch_data = self.prepare_data(*selected_batch_data)

        probability = self.model.get_policy_input(*(selected_batch_data + args))
        action = self.policy.take_action(probability, False)

        result = [index for i, index in enumerate(batch_index) if action[i]]

        self.last_probability = probability
        self.last_action = action

        return result


class TestPolicyUpdater(BatchUpdater):
    def __init__(self, model, all_data, policy, **kwargs):
        super(TestPolicyUpdater, self).__init__(model, all_data, **kwargs)
        self.policy = policy

    def start_new_epoch(self):
        super(TestPolicyUpdater, self).start_new_epoch()
        self.policy.start_new_epoch()

    def filter_batch(self, batch_index, *args):
        selected_batch_data = [data[batch_index] for data in self.all_data]
        selected_batch_data = self.prepare_data(*selected_batch_data)
        selected_batch_data.extend(args)

        probability = self.model.get_policy_input(*selected_batch_data)
        action = self.policy.take_action(probability, False)

        result = [index for i, index in enumerate(batch_index) if action[i]]

        return result


class RandomDropUpdater(BatchUpdater):
    def __init__(self, model, all_data, random_drop_number_file, **kwargs):
        self.drop_num_type = kwargs.pop('drop_num_type', 'epoch')
        self.valid_freq = kwargs.pop('valid_freq', 0)

        super(RandomDropUpdater, self).__init__(model, all_data, **kwargs)
        self.random_drop_numbers = [int(l.strip()) for l in open(random_drop_number_file, 'r')]

    def filter_batch(self, batch_index, *args):
        if self.drop_num_type == 'epoch':
            p = 1 - float(self.random_drop_numbers[self.epoch]) / self.data_size
        elif self.drop_num_type == 'vp':    # Validation point
            # Get the most nearby seen number
            i, e = 0, 0
            for i, e in enumerate(self.random_drop_numbers):
                if e >= self.total_seen_cases:
                    break
            p = 1 - float(self.valid_freq * (i + 1)) / e
        else:
            raise ValueError('Unknown drop number type {}'.format(self.drop_num_type))

        action = np.random.binomial(
            1,
            p,
            self.all_data[-1].shape
        ).astype(bool)

        return [index for i, index in enumerate(batch_index) if action[i]]
