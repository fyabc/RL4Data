#! /usr/bin/python

from __future__ import print_function, unicode_literals

import os
import json

__author__ = 'fyabc'

Config = json.load(open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r'))
ParamConfig = Config['parameters']
IMDBConfig = Config['imdb']

if __name__ == '__main__':
    print(Config)
    print(ParamConfig)
