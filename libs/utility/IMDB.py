#! /usr/bin/python

from __future__ import print_function

import cPickle as pkl

import numpy as np

from config import IMDBConfig as ParamConfig
from utils import fX, get_minibatches_idx
from my_logging import logging, message

__author__ = 'fyabc'


@logging
def load_imdb_data(data_dir=None, n_words=100000, valid_portion=0.1, maxlen=None, sort_by_len=True):
    """Loads the dataset

    :type data_dir: String
    :param data_dir: The path to the dataset (here IMDB)
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknown (1).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence length for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.
    """

    data_dir = data_dir or ParamConfig['data_dir']

    import gzip
    if data_dir.endswith(".gz"):
        f = gzip.open(data_dir, 'rb')
    else:
        f = open(data_dir, 'rb')

    train_set = pkl.load(f)
    test_set = pkl.load(f)
    f.close()

    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train_data = (train_set_x, train_set_y)
    valid_data = (valid_set_x, valid_set_y)
    test_data = (test_set_x, test_set_y)

    return train_data, valid_data, test_data


@logging
def preprocess_imdb_data(train_data, valid_data, test_data):
    train_x, train_y = train_data
    test_x, test_y = test_data

    test_size = ParamConfig['test_size']
    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = np.arange(len(test_x))
        np.random.shuffle(idx)
        idx = idx[:test_size]
        test_data = ([test_x[n] for n in idx], [test_y[n] for n in idx])

    ydim = np.max(train_y) + 1

    ParamConfig['ydim'] = ydim

    return train_data, valid_data, test_data


def prepare_imdb_data(seqs, labels, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same length: the length of the
    longest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    length.

    This swap the axis!
    """

    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples)).astype('int64')
    x_mask = np.zeros((maxlen, n_samples)).astype(fX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


###########################
# Other utilities of IMDB #
###########################

def pr(pp, name):
    return '%s_%s' % (pp, name)


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(fX)


def test():
    train_data, valid_data, test_data = load_imdb_data(maxlen=1000)
    train_x, train_y = train_data
    valid_x, valid_y = valid_data
    test_x, test_y = test_data

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    valid_x = np.asarray(valid_x)
    valid_y = np.asarray(valid_y)
    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)

    print('Train:', train_x.shape, train_y.shape)
    print('Valid:', valid_x.shape, valid_y.shape)
    print('Test:', test_x.shape, test_y.shape)

    print(sum(train_y), sum(valid_y), sum(test_y))
    

def pre_process_IMDB_data():
    # Loading data
    train_data, valid_data, test_data = load_imdb_data(n_words=ParamConfig['n_words'],
                                                       valid_portion=ParamConfig['valid_portion'],
                                                       maxlen=ParamConfig['maxlen'])
    train_data, valid_data, test_data = preprocess_imdb_data(train_data, valid_data, test_data)

    train_x, train_y = train_data
    valid_x, valid_y = valid_data
    test_x, test_y = test_data

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    valid_x = np.asarray(valid_x)
    valid_y = np.asarray(valid_y)
    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)

    train_size = len(train_x)
    valid_size = len(valid_x)
    test_size = len(test_x)

    message('Training data size:', train_size)
    message('Validation data size:', valid_size)
    message('Test data size:', test_size)

    return train_x, train_y, valid_x, valid_y, test_x, test_y, train_size, valid_size, test_size


def pre_process_config(model, train_size, valid_size, test_size):
    kf_valid = get_minibatches_idx(valid_size, model.validate_batch_size)
    kf_test = get_minibatches_idx(test_size, model.validate_batch_size)

    valid_freq = ParamConfig['valid_freq']
    if valid_freq == -1:
        valid_freq = train_size // model.train_batch_size

    save_freq = ParamConfig['save_freq']
    if save_freq == -1:
        save_freq = train_size // model.train_batch_size

    display_freq = ParamConfig['display_freq']
    save_to = ParamConfig['save_to']
    patience = ParamConfig['patience']

    return kf_valid, kf_test, valid_freq, save_freq, display_freq, save_to, patience


if __name__ == '__main__':
    test()
