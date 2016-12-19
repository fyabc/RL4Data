#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys


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
