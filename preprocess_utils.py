#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

__author__ = 'fyabc'

# The escaped string double quote.
_StringDoubleQuote = '@'
_GlobalPrefix = 'G.'
_PolicyPrefix = 'P.'
_KeyValueSeparator = '='
Tilde = '~'


def simple_parse_args2(args):
    global_args_dict = {}
    policy_args_dict = {}
    param_args_dict = {}

    for i, arg in enumerate(args):
        arg = arg.replace(_StringDoubleQuote, '"')

        if _KeyValueSeparator in arg:
            if arg.startswith(_GlobalPrefix):
                arg = arg[2:]
                the_dict = global_args_dict
            elif arg.startswith(_PolicyPrefix):
                arg = arg[2:]
                the_dict = policy_args_dict
            else:
                the_dict = param_args_dict

            key, value = arg.split(_KeyValueSeparator)
            the_dict[key] = eval(value)
        else:
            if i > 0:
                print('Warning: The argument {} is unused'.format(arg))

    return global_args_dict, policy_args_dict, param_args_dict


def check_config(param_config, policy_config):
    pass


def strict_update(target, new_dict):
    for k, v in new_dict.iteritems():
        if k not in target:
            raise KeyError('The key {} is not in the parameters.'.format(k))
        target[k] = v
