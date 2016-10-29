#! /usr/bin/python

from __future__ import print_function, unicode_literals

import os
import json
import re

__author__ = 'fyabc'

# Load JSON config file and remove line comments
_lines = list(open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r'))

for i, line in enumerate(_lines):
    _lines[i] = re.sub(r'//.*\n', '\n', line)

Config = json.loads(''.join(_lines))
CifarConfig = Config['cifar10']
IMDBConfig = Config['imdb']
MNISTConfig = Config['mnist']
PolicyConfig = Config['policy']

if __name__ == '__main__':
    print(Config)
    print(CifarConfig)
