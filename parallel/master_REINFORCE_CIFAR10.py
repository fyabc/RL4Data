#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from parallel_utils import get_gpu_id


def main():
    gpu_ids, remain_args = get_gpu_id()


if __name__ == '__main__':
    main()
