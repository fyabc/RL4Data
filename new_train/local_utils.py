# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys
import os

__author__ = 'fyabc'


def get_project_root_path():
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    return root_path


def _test():
    get_project_root_path()


if __name__ == '__main__':
    _test()
