#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import numpy as np
import theano.tensor as T

from .utils import fX, f_open

__author__ = 'fyabc'


Profile = False


def p_(prefix, name):
    """Get the name of tensor with the prefix (layer name) and variable name."""
    return '{}_{}'.format(prefix, name)


# Parameter initializer
def orthogonal_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, _, _ = np.linalg.svd(W)

    return u.astype(fX)


def normal_weight(n_in, n_out=None, scale=0.01, orthogonal=True):
    n_out = n_in if n_out is None else n_out

    if n_in == n_out and orthogonal:
        W = orthogonal_weight(n_in)
    else:
        W = scale * np.random.randn(n_in, n_out)
    return W.astype(fX)


# Some other utils
def concatenate(tensors, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Back-propagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> import theano.tensor as T
        >>> x, y = T.matrices('x', 'y')
        >>> concatenate([x, y], axis=1)
        IncSubtensor{Set;::, int64:int64:}.0

    :parameters:
        - tensors : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """

    concat_size = sum(t.shape[axis] for t in tensors)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensors[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensors[0].ndim):
        output_shape += (tensors[0].shape[k],)

    out = T.zeros(output_shape)
    offset = 0
    for t in tensors:
        indices = [slice(None) for _ in range(axis)] + [slice(offset, offset + t.shape[axis])] + \
                  [slice(None) for _ in range(axis + 1, tensors[0].ndim)]

        out = T.set_subtensor(out[indices], t)
        offset += t.shape[axis]

    return out


tanh = T.tanh


def linear(x): return x


class TextIterator(object):
    """The text iterator of NMT input data."""

    UNK = 1

    def __init__(self, source, target,
                 source_dict, target_dict,
                 batch_size=128, maxlen=100,
                 n_words_source=-1, n_words_target=-1):
        self.source = f_open(source, mode='r', unpickle=False)
        self.target = f_open(target, mode='r', unpickle=False)
        self.source_dict = f_open(source_dict, mode='rb', unpickle=True)
        self.target_dict = f_open(target_dict, mode='rb', unpickle=True)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * 40

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)
        self.end_of_data = False

    def next(self):
        if self.end_of_data:
            self.reset()
            raise StopIteration

        source = []
        target = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if not self.source_buffer:
            for k_ in xrange(self.k):
                s = self.source.readline()
                if s == '':
                    break
                t = self.target.readline()
                if t == '':
                    break

                self.source_buffer.append(s.strip().split())
                self.target_buffer.append(t.strip().split())

            # sort by target buffer
            t_len = np.array([len(t) for t in self.target_buffer])
            t_idx = t_len.argsort()

            self.source_buffer = [self.source_buffer[i] for i in t_idx]
            self.target_buffer = [self.target_buffer[i] for i in t_idx]

        if not self.source_buffer or not self.target_buffer:
            self.reset()
            raise StopIteration

        try:
            # actual work here
            while True:
                # read from source file and map to word index
                if not self.source_buffer:
                    break

                s = self.source_buffer.pop()
                s = (self.source_dict.get(w, self.UNK) for w in s)

                if self.n_words_source > 0:
                    s = [w if w < self.n_words_source else 1 for w in s]
                else:
                    s = list(s)

                t = self.target_buffer.pop()
                t = (self.target_dict.get(w, self.UNK) for w in t)

                if self.n_words_target > 0:
                    t = [w if w < self.n_words_target else 1 for w in t]
                else:
                    s = list(t)

                if len(s) > self.maxlen and len(t) > self.maxlen:
                    continue

                source.append(s)
                target.append(t)

                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break

        except IOError:
            self.end_of_data = True

        if not source or not target:
            self.reset()
            raise StopIteration

        return np.array(source, dtype='int64'), np.array(target, dtype='int64')


def prepare_NMT_data(xs, ys, maxlen=None):
    """Batch preparation of NMT data.

    This swap the axis!

    Parameters
    ----------
    xs: a list of source sentences
    ys: a list of target sentences
    maxlen: max length of sentences.

    Returns
    -------
    x, x_mask, y, y_mask: numpy arrays (maxlen * n_samples)
    """

    x_lens = [len(s) for s in xs]
    y_lens = [len(s) for s in ys]

    # Filter long sentences.
    if maxlen is not None:
        xs_new, ys_new = [], []
        x_lens_new, y_lens_new = [], []

        for lx, sx, ly, sy in zip(x_lens, xs, y_lens, ys):
            if lx < maxlen and ly < maxlen:
                xs_new.append(sx)
                x_lens_new.append(lx)
                ys_new.append(sy)
                y_lens_new.append(ly)

        xs, x_lens, ys, y_lens = xs_new, x_lens_new, ys_new, y_lens_new

        if not x_lens or not y_lens:
            return None, None, None, None

    n_samples = len(xs)
    maxlen_x = np.max(x_lens) + 1
    maxlen_y = np.max(y_lens) + 1

    x = np.zeros((maxlen_x, n_samples), dtype='int64')
    y = np.zeros((maxlen_y, n_samples), dtype='int64')
    x_mask = np.zeros((maxlen_x, n_samples), dtype=fX)
    y_mask = np.zeros((maxlen_y, n_samples), dtype=fX)

    for i, (sx, sy) in enumerate(zip(xs, ys)):
        x[:x_lens[i], i] = sx
        x_mask[:x_lens[i] + 1, i] = 1.
        y[:y_lens[i], i] = sy
        y_mask[:y_lens[i] + 1, i] = 1.

    return x, x_mask, y, y_mask
