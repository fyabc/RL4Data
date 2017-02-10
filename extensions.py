#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import numpy as np

from logging_utils import message


class PartLossChecker(object):
    def __init__(self,
                 updater,
                 check_parts=('range(20000, 30000)', 'range(0, 20000) + range(30000, 50000)'),
                 check_per_epoch=2
                 ):
        self.updater = updater
        self.check_parts = check_parts
        self.check_freq = self.updater.data_size // self.updater.batch_size // check_per_epoch

    def check(self):
        if self.updater.epoch_train_batches % self.check_freq == 0:
            inputs, targets = self.updater.all_data

            message('Check point: epoch {} batch {}'.format(self.updater.epoch, self.updater.epoch_train_batches))
            for check_part_str in self.check_parts:
                check_part = eval(check_part_str)
                losses = self.updater.model.f_cost_list_without_decay(
                    *(data[check_part] for data in self.updater.all_data))

                # Get margins
                probabilities = self.updater.model.f_probs(inputs[check_part])
                margins = np.zeros((len(check_part),))
                for i, prob in enumerate(probabilities):
                    margins[i] = prob[targets[check_part[i]]]
                    prob[targets[check_part[i]]] = -np.inf
                    margins[i] -= np.max(prob)

                message('''\
    Part {}:
        Loss: mean={}, std={}
        Margin: mean={}, std={}
'''.format(check_part_str, losses.mean(), losses.std(), margins.mean(), margins.std()))
