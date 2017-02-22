#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from collections import deque
import heapq
from itertools import izip

import numpy as np

from extensions import PartLossChecker
from config import Config, PolicyConfig
from utils import message, get_rank

# Some Magic Numbers.

# Number of classes and size of each class.
ClassSize = 5000
ClassNumber = 10
TotalSize = ClassSize * ClassNumber

# Compute loss when needed in "log_data" (all updaters except SPL).
# [NOTE] It will cause computation of loss on each data, and will slow down the training process.
ComputeLoss = True

# The target distribution of each corrupt levels.
# [NOTE] This can be changed.
TargetDistribution = np.array([0.20, 0.16, 0.12, 0.08, 0.04, 0.00, 0.04, 0.08, 0.12, 0.16])


def _score(distribution):
    return np.sum((TargetDistribution - distribution) ** 2)


class BatchUpdater(object):
    def __init__(self, model, all_data, **kwargs):
        """

        Parameters
        ----------
        model
        all_data
        kwargs :
            prepare_data: function, optional
                The prepare data function.
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

        # Validate point number
        self.vp_number = 0

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

        self.history_accuracy = []

        if Config['temp_job'] == 'log_data':
            # self.part_updated_indices = [0 for _ in range(ClassNumber)]
            # self.updated_indices = [0 for _ in range(ClassNumber)]

            self.part_avg_loss = [[] for _ in range(ClassNumber)]
            self.avg_loss = [[] for _ in range(ClassNumber)]

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
                (Unused now)
                It will be append to the data list.
            The arguments for get_policy_input:
                *data_list, updater, history_accuracy, *args

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

        # Get x[], mask[], y[], ..., then prepare them
        selected_batch_data = [data[self.last_update_batch_index] for data in self.all_data]

        if Config['temp_job'] == 'check_selected_data_label':
            selected_batch_label = selected_batch_data[-1]
            for i in range(len(self.epoch_label_count)):
                count_i = sum(selected_batch_label == i)
                self.epoch_label_count[i] += count_i
                self.total_label_count[i] += count_i

        # [NOTE] Prepared data may swap the axis (in IMDB)!
        p_selected_batch_data = self.prepare_data(*selected_batch_data)
        part_train_cost = self.model.f_train(*p_selected_batch_data)

        if np.isinf(part_train_cost) or np.isnan(part_train_cost):
            raise OverflowError('NaN detected at epoch {} case {}'.format(self.epoch, self.epoch_accepted_cases))

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

    # Only for temp_job:"log_data"
    def add_index(self, index, loss=0.0):
        # [NOTE] cifar10 have mirrored data, so mod TotalSize.
        clazz = (index % TotalSize) // ClassSize

        # self.part_updated_indices[clazz] += 1
        # self.updated_indices[clazz] += 1

        self.part_avg_loss[clazz].append(loss)
        self.avg_loss[clazz].append(loss)

    # Only for temp_job:"log_data"
    def add_index_list(self, indices):
        if Config['temp_job'] == 'log_data':
            if ComputeLoss:
                selected_batch_data = [data[indices] for data in self.all_data]
                selected_batch_data = self.prepare_data(*selected_batch_data)
                cost_list = self.model.f_cost_list_without_decay(*selected_batch_data)
            else:
                cost_list = [0.0 for _ in indices]
            for idx, loss in izip(indices, cost_list):
                self.add_index(idx, loss)

    # Only for temp_job:"log_data"
    def log_data_message_at_vp(self, reset=True, test_margin=True):
        updated_indices = [len(loss_list) for loss_list in self.avg_loss]
        part_updated_indices = [len(loss_list) for loss_list in self.part_avg_loss]

        total_indices = sum(updated_indices)
        part_total_indices = sum(part_updated_indices)

        part_ratios = [n / float(part_total_indices) for n in part_updated_indices]
        total_ratios = [n / float(total_indices) for n in updated_indices]

        message('[Log Data]')
        message('Part  (total {:>8}): {}'.format(
            part_total_indices,
            '\t'.join(format(val, '.3f') for val in part_ratios)))
        message('Whole (total {:>8}): {}'.format(
            total_indices,
            '\t'.join(format(val, '.3f') for val in total_ratios)))
        message('Score                 :',
                'Part =', format(_score(part_ratios), '.6f'),
                'Total = ', format(_score(total_ratios), '.6f'))

        message('Part Avg Loss         :', '\t'.join(format(np.mean(l_list), '.3f') for l_list in self.part_avg_loss))
        message('Part Loss Std         :', '\t'.join(format(np.std(l_list), '.3f') for l_list in self.part_avg_loss))
        message('Total Avg Loss        :', '\t'.join(format(np.mean(l_list), '.3f') for l_list in self.avg_loss))
        message('Total Loss Std        :', '\t'.join(format(np.std(l_list), '.3f') for l_list in self.avg_loss))

        if test_margin:
            # Test to get the (average) margin and loss rank.
            # Get 10 batches in each corrupt level.
            SampleBatchNumber = 100

            message('Test the margin for all {} corrupt levels:'.format(ClassNumber))

            mean_feature_list = []
            std_feature_list = []

            for level in range(ClassNumber):
                class_range = np.arange(level * ClassSize, (level + 1) * ClassSize)

                to_be_stacked = []

                for _ in range(SampleBatchNumber):
                    sample_batch_idx = np.random.choice(class_range, self.batch_size)
                    sample_batch_data = [data[sample_batch_idx] for data in self.all_data]
                    p_sample_batch_data = self.prepare_data(*sample_batch_data)

                    features = self.model.get_policy_input(*(p_sample_batch_data + (self, self.history_accuracy)))
                    to_be_stacked.append(features)

                all_features = np.hstack(to_be_stacked)

                mean_features = all_features.mean(axis=0)
                std_features = all_features.std(axis=0)

                mean_feature_list.append(mean_features)
                std_feature_list.append(std_features)

            message('Margin Avg            :', '\t'.join(format(mf[-4], '.3f') for mf in mean_feature_list))
            message('Margin Std            :', '\t'.join(format(sf[-4], '.3f') for sf in std_feature_list))

        message('[Log Data End]')

        if reset:
            # self.part_updated_indices = [0 for _ in range(ClassNumber)]
            self.part_avg_loss = [[] for _ in range(ClassNumber)]

    def log_dropped_data_message_at_vp(self):
        pass


class RawUpdater(BatchUpdater):
    def __init__(self, model, all_data, **kwargs):
        super(RawUpdater, self).__init__(model, all_data, **kwargs)

    def filter_batch(self, batch_index, *args):
        self.add_index_list(batch_index)
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
                            self.add_index(batch_index[j], cost_list[j])

        return result


class TrainPolicyUpdater(BatchUpdater):
    def __init__(self, model, all_data, policy, **kwargs):
        super(TrainPolicyUpdater, self).__init__(model, all_data, **kwargs)
        self.policy = policy

    def start_new_epoch(self):
        super(TrainPolicyUpdater, self).start_new_epoch()
        # self.policy.start_new_validation_point()

    def filter_batch(self, batch_index, *args):
        selected_batch_data = [data[batch_index] for data in self.all_data]
        selected_batch_data = self.prepare_data(*selected_batch_data)

        probability = self.model.get_policy_input(*(selected_batch_data + (self, self.history_accuracy) + args))
        action = self.policy.take_action(probability, True)

        result = [index for i, index in enumerate(batch_index) if action[i]]

        self.add_index_list(result)

        return result


class ACUpdater(BatchUpdater):
    def __init__(self, model, all_data, policy, **kwargs):
        super(ACUpdater, self).__init__(model, all_data, **kwargs)
        self.policy = policy

        self.last_probability = None
        self.last_action = None

    def start_new_epoch(self):
        super(ACUpdater, self).start_new_epoch()
        self.policy.start_new_validation_point()

    def filter_batch(self, batch_index, *args):
        selected_batch_data = [data[batch_index] for data in self.all_data]
        selected_batch_data = self.prepare_data(*selected_batch_data)

        probability = self.model.get_policy_input(*(selected_batch_data + (self, self.history_accuracy) + args))
        action = self.policy.take_action(probability, False)

        result = [index for i, index in enumerate(batch_index) if action[i]]

        self.add_index_list(result)

        self.last_probability = probability
        self.last_action = action

        return result


class TestPolicyUpdater(BatchUpdater):
    def __init__(self, model, all_data, policy, **kwargs):
        super(TestPolicyUpdater, self).__init__(model, all_data, **kwargs)
        self.policy = policy

        if Config['temp_job'] == 'log_dropped_data':
            self.total_dropped_ranks = [0 for _ in range(self.batch_size)]
            self.part_dropped_ranks = [0 for _ in range(self.batch_size)]

    def start_new_epoch(self):
        super(TestPolicyUpdater, self).start_new_epoch()
        self.policy.start_new_validation_point()

    def filter_batch(self, batch_index, *args):
        selected_batch_data = [data[batch_index] for data in self.all_data]
        p_selected_batch_data = self.prepare_data(*selected_batch_data)

        probability = self.model.get_policy_input(*(p_selected_batch_data + (self, self.history_accuracy) + args))
        action = self.policy.take_action(probability, False)

        result = [index for i, index in enumerate(batch_index) if action[i]]

        # todo: log features of dropped data here
        if Config['temp_job'] == 'log_dropped_data':
            # [NOTE] Just log rank now.
            if PolicyConfig['add_loss_rank'] is True:
                for a, p in izip(action, probability):
                    if not a:
                        rank = int(round(p[-2] * self.batch_size))

                        self.total_dropped_ranks[rank] += 1
                        self.part_dropped_ranks[rank] += 1
                else:
                    # [NOTE] Compatibility for old IMDB version
                    # loss = -log(P(y)), output of PolicyConfig['add_label_input'] is log(P(y)).
                    losses = np.array([-prob[self.model.output_size] for prob in probability])

                    rank = get_rank(losses)

                    for a, r in izip(action, rank):
                        if not a:
                            self.total_dropped_ranks[r] += a
                            self.part_dropped_ranks[r] += a

        self.add_index_list(result)

        return result

    def log_dropped_data_message_at_vp(self):
        message('[Log Dropped Data]')
        message('Rank distribution (0 -> batch_size) of dropped data')

        message('Part  (total {:>8}):'.format(sum(self.part_dropped_ranks)),
                '\t'.join(str(r) for r in self.part_dropped_ranks))
        message('Total (total {:>8}):'.format(sum(self.total_dropped_ranks)),
                '\t'.join(str(r) for r in self.total_dropped_ranks))

        message('[Log Dropped Data End]')

        self.part_dropped_ranks = [0 for _ in range(self.batch_size)]


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
            if i == 0:
                prev_e = 0
            else:
                prev_e = self.random_drop_numbers[i - 1]
            # Get the ratio of this valid point
            p = float(self.valid_freq * self.batch_size) / (e - prev_e)
        else:
            raise ValueError('Unknown drop number type {}'.format(self.drop_num_type))

        action = np.random.binomial(
            1,
            p,
            self.all_data[-1].shape
        ).astype(bool)

        return [index for i, index in enumerate(batch_index) if action[i]]
