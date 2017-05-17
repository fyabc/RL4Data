# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import fnmatch


def get_path(base_path, dataset_name, filename=None):
    """Get the dataset specific path.

    Parameters
    ----------
    base_path: DataPath, LogPath, etc.
    dataset_name: cifar10, mnist, etc.
    filename: optional, the file name.

    Returns
    -------
    The joined path.
    """

    if filename is None:
        return os.path.join(base_path, dataset_name)
    return os.path.join(base_path, dataset_name, filename)


def split_policy_name(policy_name):
    tmp, ext = os.path.splitext(policy_name)
    name, episode = os.path.splitext(tmp)

    return name, episode, ext


def find_newest(dir_name, raw_name, ext='.npz', ret_number=False):
    max_number = -1
    newest_filename = ''

    pattern = '{}.*{}'.format(os.path.basename(raw_name), ext)

    for filename in os.listdir(dir_name):
        if fnmatch.fnmatch(filename, pattern):
            name, episode, ext = split_policy_name(filename)

            try:
                episode = int(episode[1:])
            except ValueError:
                continue

            if episode > max_number:
                max_number = episode
                newest_filename = filename

    if newest_filename:
        newest_filename = os.path.join(dir_name, newest_filename)

    if ret_number:
        return newest_filename, max_number
    return newest_filename
