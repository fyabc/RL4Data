#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
from functools import wraps

from config import Config

# Logging settings
logging_file = sys.stderr
_depth = 0


def init_logging_file(append=False, new=True):
    """

    Parameters
    ----------
    append: bool
        Append exist logging file?
    new: bool
        Create new logging file? If False, will overwrite old logging file.
    Returns
    -------

    """

    global logging_file

    if Config['logging_file'] is None:
        return

    if append:
        logging_file = open(Config['logging_file'], 'a')
        return

    raw_filename = Config['logging_file']
    i = 1

    filename = raw_filename

    if new:
        while os.path.exists(filename):
            filename = raw_filename.replace('.txt', '_{}.txt'.format(i))
            i += 1

    Config['logging_file'] = filename
    logging_file = open(filename, 'w')


def finalize_logging_file():
    if logging_file != sys.stderr:
        logging_file.flush()
        logging_file.close()


def get_logging_file():
    global logging_file
    return logging_file


def message(*args, **kwargs):
    if logging_file != sys.stderr:
        print(*args, file=logging_file, **kwargs)
    print(*args, file=sys.stderr, **kwargs)


def logging(func, file_=sys.stderr):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _depth

        message(' ' * 2 * _depth + '[Start function %s...]' % func.__name__)
        _depth += 1
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        _depth -= 1
        message(' ' * 2 * _depth + '[Function %s done, time: %.3fs]' % (func.__name__, end_time - start_time))
        return result
    return wrapper
