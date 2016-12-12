#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys
import theano
import time
import random

__author__ = 'fyabc'

sleep_time = random.uniform(0, 3)

time.sleep(sleep_time)

print('Echo from slave, sleep {:.6}s'.format(sleep_time))
print('FloatX:', theano.config.floatX)
print('Error:', file=sys.stderr)
