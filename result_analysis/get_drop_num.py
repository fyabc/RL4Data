#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import os

from utils import load_list, save_list


DropNumPath = './drop_num'


def main(**kwargs):
    src = kwargs.pop('src', 'drop_num_flip_speed_best_base.txt')
    tgt = kwargs.pop('tgt', 'drop_num_flip_speed_best.txt')

    l = load_list(os.path.join(DropNumPath, src))


if __name__ == '__main__':
    # argparse_main()
    main()
