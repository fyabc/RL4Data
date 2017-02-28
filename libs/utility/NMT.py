#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import numpy as np
import theano.tensor as T

from .utils import fX

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
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (spliting) needs to be done on the CPU.
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

    pass
