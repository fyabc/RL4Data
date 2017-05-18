#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import os
import sys
import argparse

ProjectRootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ProjectRootPath)

from libs.utility.config import LogPath

__author__ = 'fyabc'


StartTags = {
    'test_acc': ['Test accuracy:', 'TeA:'],
    'valid_acc': ['Valid accuracy:', 'VA:'],
}


def find(args):
    best_number = -1000000
    best_filename = ''
    best_line_no = -1

    start_tags = StartTags[args.name]

    if args.file is not None:
        filenames = [args.file]
    else:
        filenames = os.listdir(os.path.join(LogPath, args.dataset))

    for filename in filenames:
        if not filename.startswith('log-{}-{}'.format(args.dataset, args.type)):
            continue

        with open(os.path.join(LogPath, args.dataset, filename), 'r') as f:
            for i, line in enumerate(f):
                for tag in start_tags:
                    if line.startswith(tag):
                        value = float(line.split()[-1])
                        if value > best_number:
                            best_number = value
                            best_filename = filename
                            best_line_no = i + 1
                        continue

    if args.file is not None:
        print("Best '{}' in '{}' file:".format(args.name, args.file))
    else:
        print("Best '{}' in '{}' type of '{}' dataset:".format(args.name, args.type, args.dataset))
    print("{} in line {} of file '{}'".format(best_number, best_line_no, best_filename))


def main(args=None):
    parser = argparse.ArgumentParser(description='Find the best number')
    parser.add_argument('-d', '--dataset', action='store', dest='dataset', default='imdb',
                        help='The dataset (default is "%(default)s")')
    parser.add_argument('-n', '--name', action='store', dest='name', default='test_acc',
                        help='The find name (default is "%(default)s")')
    parser.add_argument('-t', '--type', action='store', dest='type', default='raw',
                        help='The target job type (default is %(default)s)')
    parser.add_argument('-f', '--file', action='store', dest='file', default=None,
                        help='Find in specific file (default is %(default)s)')

    options = parser.parse_args(args)

    find(options)


if __name__ == '__main__':
    # main(['-t', 'raw'])
    # main(['-t', 'stochastic'])
    # main(['-t', 'random_drop'])
    # main(['-t', 'spl'])
    # main(['-t', 'raw', '-d', 'cifar10'])
    # main(['-t', 'stochastic', '-d', 'cifar10'])
    # main(['-t', 'random_drop', '-d', 'cifar10'])
    # main(['-t', 'spl', '-d', 'cifar10'])
    # main(['-t', 'reinforce', '-d', 'imdb', '-n', 'valid_acc', '-f', 'log-imdb-reinforce-lr-best_acc-W.txt'])
    # main(['-t', 'reinforce', '-d', 'imdb', '-n', 'valid_acc', '-f', 'log-imdb-reinforce-lr-best_acc-RB.txt'])
    # main(['-t', 'reinforce', '-d', 'imdb', '-n', 'valid_acc', '-f', 'log-imdb-reinforce-lr-best_acc-NoRB.txt'])
    # main(['-t', 'reinforce', '-d', 'imdb', '-n', 'valid_acc', '-f', 'log-imdb-reinforce-lr-best_acc-b0.txt'])
    # main(['-t', 'raw', '-d', 'imdb', '-n', 'valid_acc'])
    # main(['-t', 'spl', '-d', 'imdb', '-n', 'valid_acc'])
    # main(['-t', 'stochastic', '-d', 'imdb', '-n', 'valid_acc'])
    # main(['-t', 'random_drop', '-d', 'imdb', '-n', 'valid_acc'])
    main()
