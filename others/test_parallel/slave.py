#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import sys
import theano
import time
import random

sleep_time = random.uniform(0, 3)

time.sleep(sleep_time)

print('Echo from slave, sleep {:.6}s'.format(sleep_time))
print('FloatX:', theano.config.floatX)
print('Error:', file=sys.stderr)
