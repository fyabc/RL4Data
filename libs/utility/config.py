#! /usr/bin/python

from __future__ import print_function

import os
import json
import re

__author__ = 'fyabc'

# Paths
ProjectRootPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DataPath = os.path.join(ProjectRootPath, 'data')
LogPath = os.path.join(ProjectRootPath, 'log')
ModelPath = os.path.join(ProjectRootPath, 'model')
ReservedDataPath = os.path.join(ProjectRootPath, 'reserved_data')

# Load JSON config file and remove line comments
_lines = list(open(os.path.join(ProjectRootPath, 'config.json'), 'r'))

for _i, line in enumerate(_lines):
    _lines[_i] = re.sub(r'//.*\n', '\n', line)

Config = json.loads(''.join(_lines))
C = Config

PolicyConfig = Config['policy']
PC = PolicyConfig

CifarConfig = Config['cifar10']
IMDBConfig = Config['imdb']
MNISTConfig = Config['mnist']
NMTConfig = Config['nmt']

# All train types.
TrainTypes = {
    'raw',
    'self_paced',
    'spl',
    'policy',
    'reinforce',
    'speed',
    'actor_critic',
    'ac',
    'deterministic',
    'stochastic',
    'random_drop',
    'new_train',
}

# Some train types sets.
NoPolicyTypes = {
    'raw',
    'self_paced',
    'spl',
    'random_drop',
}

CommonTypes = {
    'deterministic',
    'stochastic',
}

ReinforceTypes = {
    'policy',
    'reinforce',
    'speed',
}


if __name__ == '__main__':
    print(Config)
    print(CifarConfig)
