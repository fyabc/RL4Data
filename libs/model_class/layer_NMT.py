#! /usr/bin/python
# -*- encoding: utf-8 -*-

"""Some layers used in NMT model."""

from __future__ import print_function, unicode_literals

import numpy as np
import theano
import theano.tensor as T

from ..utility.config import NMTConfig as ParamConfig
from ..utility.utils import fX
from ..utility.NMT import p_, normal_weight, orthogonal_weight, Profile

__author__ = 'fyabc'


# Some utilities.

# utility function to slice a tensor
def _slice(_x, n, _dim):
    if _x.ndim == 3:
        return _x[:, :, n * _dim:(n + 1) * _dim]
    return _x[:, n * _dim:(n + 1) * _dim]


# Function of layers and their initializers.
# [NOTE]
# Layer
# The 1st argument (in common) is input_: a Theano tensor that represent the input.
# The 2nd argument (in common) is params: the dict of (tensor) parameters.
# The 3rd argument (in common) is prefix: the prefix layer name.
#
# Initializer
# The 1st argument (in common) is params: the dict of (numpy) parameters.
# The 2nd argument (in common) is prefix: the prefix layer name.
# Optional argument n_in: input size.
# Optional argument n_out: output size.
# Optional argument dim: dimension size (hidden size?).
#
# These functions return another Theano tensor as its output.

def dropout(input_, use_noise, rand):
    return T.switch(
        use_noise,
        input_ * rand.binomial(input_.shape, p=0.5, n=1, dtype=fX),
        input_ * 0.5
    )


# feed-forward layer: affine transformation + point-wise nonlinearity
def feed_forward(input_, params, prefix='rconv', activation=T.tanh, **kwargs):
    return activation(T.dot(input_, params[p_(prefix, 'W')]) + params[p_(prefix, 'b')])


def init_feed_forward(params, prefix='ff', n_in=None, n_out=None, orthogonal=True):
    n_in = ParamConfig['dim_proj'] if n_in is None else n_in
    n_out = ParamConfig['dim_proj'] if n_out is None else n_out

    params[p_(prefix, 'W')] = normal_weight(n_in, n_out, scale=0.01, orthogonal=orthogonal)
    params[p_(prefix, 'b')] = np.zeros((n_out,), dtype=fX)

    return params


# GRU layer
def gru(input_, params, prefix='gru', mask=None, **kwargs):
    n_steps = input_.shape[0]
    n_samples = input_.shape[1] if input_.ndim == 3 else 1
    dim = params[p_(prefix, 'Ux')].shape[1]
    mask = T.alloc(1., n_steps, 1) if mask is None else mask

    # input_ is the input word embeddings
    # input to the gates, concatenated
    input_g = T.dot(input_, params[p_(prefix, 'W')]) + params[p_(prefix, 'b')]

    # input to compute the hidden state proposal
    input_x = T.dot(input_, params[p_(prefix, 'Wx')]) + params[p_(prefix, 'bx')]

    # step function to be used by scan
    # args   : sequences          |outputs| non-seqs
    def _step(mask, input, input_x, hidden, U, Ux):
        p_react = T.dot(hidden, U) + input

        # reset and update gates
        r = T.nnet.sigmoid(_slice(p_react, 0, dim))
        u = T.nnet.sigmoid(_slice(p_react, 1, dim))

        # compute the hidden state proposal
        p_react_x = T.dot(hidden, Ux) * r + input_x

        # hidden state proposal
        h = T.tanh(p_react_x)

        # leaky integrate and obtain next hidden state
        h = u * hidden + (1. - u) * h
        h = mask[:, None] * h + (1. - mask)[:, None] * hidden

        return h

    # prepare scan arguments
    seqs = [mask, input_g, input_x]
    init_states = [T.alloc(0., n_samples, dim)]
    shared_vars = [params[p_(prefix, 'U')], params[p_(prefix, 'Ux')]]

    result, _ = theano.scan(
        _step,
        sequences=seqs,
        outputs_info=init_states,
        non_sequences=shared_vars,
        name=p_(prefix, '_layers'),
        n_steps=n_steps,
        profile=Profile,
        strict=True,
    )

    return result


def init_gru(params, prefix='gru', n_in=None, dim=None):
    n_in = ParamConfig['dim_proj'] if n_in is None else n_in
    dim = ParamConfig['dim_proj'] if dim is None else dim

    # embedding to gates transformation weights, biases
    params[p_(prefix, 'W')] = np.concatenate([normal_weight(n_in, dim), normal_weight(n_in, dim)], axis=1)
    params[p_(prefix, 'b')] = np.zeros((2 * dim,), dtype=fX)

    # recurrent transformation weights for gates
    params[p_(prefix, 'U')] = np.concatenate([orthogonal_weight(dim), orthogonal_weight(dim)], axis=1)

    # embedding to hidden state proposal weights, biases
    params[p_(prefix, 'Wx')] = normal_weight(n_in, dim)
    params[p_(prefix, 'bx')] = np.zeros((dim,), dtype=fX)

    # recurrent transformation weights for hidden state proposal
    params[p_(prefix, 'Ux')] = orthogonal_weight(dim)

    return params


# Conditional GRU layer with Attention
def gru_cond(input_, params, prefix='gru_cond', mask=None, **kwargs):
    context = kwargs.pop('context', None)
    one_step = kwargs.pop('one_step', False)
    init_memory = kwargs.pop('init_memory', None)
    init_state = kwargs.pop('init_state', None)
    context_mask = kwargs.pop('context_mask', None)

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    n_steps = input_.shape[0]
    n_samples = input_.shape[1] if input_.ndim == 3 else 1
    dim = params[p_(prefix, 'Wcx')].shape[1]
    mask = T.alloc(1., n_steps, 1) if mask is None else mask

    # initial/previous state
    init_state = T.alloc(0., n_samples, dim) if init_state is None else init_state

    # projected context
    assert context.ndim == 3, 'Context must be 3-d: #annotation * #sample * dim'

    p_ctx = T.dot(context, params[p_(prefix, 'Wc_att')]) + params[p_(prefix, 'b_att')]

    # projected x
    input_g = T.dot(input_, params[p_(prefix, 'W')]) + params[p_(prefix, 'b')]
    input_x = T.dot(input_, params[p_(prefix, 'Wx')]) + params[p_(prefix, 'bx')]

    def _step(mask, input_, input_x,
              hidden, ctx_, alpha_,
              p_ctx, cc_, U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx, U_nl, Ux_nl, b_nl, bx_nl
              ):
        p_react1 = T.nnet.sigmoid(T.dot(hidden, U) + input_)

        r1 = _slice(p_react1, 0, dim)
        u1 = _slice(p_react1, 1, dim)

        p_react_x1 = T.dot(hidden, Ux) * r1 + input_x
        h1 = T.tanh(p_react_x1)

        h1 = u1 * hidden + (1. - u1) * h1
        h1 = mask[:, None] * h1 + (1. - mask)[:, None] * hidden

        # attention
        p_state = T.dot(h1, W_comb_att)

        p_ctx_ = T.tanh(p_ctx + p_state[None, :, :])
        alpha = T.dot(p_ctx_, U_att) + c_tt
        alpha = T.exp(alpha.reshape([alpha.shape[0], alpha.shape[1]]))

        if context_mask:
            alpha *= context_mask

        alpha /= alpha.sum(0, keepdims=True)

        # Current context
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)

        p_react2 = T.nnet.sigmoid(T.dot(h1, U_nl) + b_nl + T.dot(ctx_, Wc))

        r2 = _slice(p_react2, 0, dim)
        u2 = _slice(p_react2, 1, dim)

        p_react_x2 = (T.dot(h1, Ux_nl) + bx_nl) * r2 + T.dot(ctx_, Wcx)

        h2 = T.tanh(p_react_x2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = mask[:, None] * h2 + (1. - mask)[:, None] * h1

        return h2, ctx_, alpha.T

    # prepare scan arguments
    seqs = [mask, input_g, input_x]
    shared_vars = [
        params[p_(prefix, name)]
        for name in ('U', 'Wc', 'W_comb_att', 'U_att', 'c_tt', 'Ux', 'Wcx', 'U_nl', 'Ux_nl', 'b_nl', 'bx_nl')
    ]

    if one_step:
        result = _step(*(seqs + [init_state, None, None, p_ctx, context] + shared_vars))
    else:
        result, _ = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=[
                init_state,
                T.alloc(0., n_samples, context.shape[2]),
                T.alloc(0., n_samples, context.shape[0]),
            ],
            non_sequences=[p_ctx, context] + shared_vars,
            name=p_(prefix, '_layers'),
            n_steps=n_steps,
            profile=Profile,
            strict=True,
        )

    return result


def init_gru_cond(params, prefix='gru_cond',
                  n_in=None, dim=None, dim_ctx=None, n_in_nonlinearity=None, dim_nonlinearity=None):
    n_in = ParamConfig['dim'] if n_in is None else n_in
    dim = ParamConfig['dim'] if dim is None else dim
    dim_ctx = ParamConfig['dim'] if dim_ctx is None else dim_ctx
    n_in_nonlinearity = n_in if n_in_nonlinearity is None else n_in_nonlinearity
    dim_nonlinearity = dim if dim_nonlinearity is None else dim_nonlinearity

    params[p_(prefix, 'W')] = np.concatenate([normal_weight(n_in, dim), normal_weight(n_in, dim)], axis=1)
    params[p_(prefix, 'b')] = np.zeros((2 * dim,), dtype=fX)
    params[p_(prefix, 'U')] = np.concatenate(
        [orthogonal_weight(dim_nonlinearity), orthogonal_weight(dim_nonlinearity)], axis=1)

    params[p_(prefix, 'Wx')] = normal_weight(n_in_nonlinearity, dim_nonlinearity)
    params[p_(prefix, 'Ux')] = orthogonal_weight(dim_nonlinearity)
    params[p_(prefix, 'bx')] = np.zeros((dim_nonlinearity,), dtype=fX)

    params[p_(prefix, 'U_nl')] = np.concatenate(
        [orthogonal_weight(dim_nonlinearity), orthogonal_weight(dim_nonlinearity)], axis=1)
    params[p_(prefix, 'b_nl')] = np.zeros((2 * dim_nonlinearity,), dtype=fX)

    params[p_(prefix, 'Ux_nl')] = orthogonal_weight(dim_nonlinearity)
    params[p_(prefix, 'bx_nl')] = np.zeros((dim_nonlinearity,), dtype=fX)

    # context to LSTM
    params[p_(prefix, 'Wc')] = normal_weight(dim_ctx, dim * 2)
    params[p_(prefix, 'Wcx')] = normal_weight(dim_ctx, dim_ctx)

    # attention: combined -> hidden
    params[p_(prefix, 'W_comb_att')] = normal_weight(dim, dim_ctx)

    # attention: context -> hidden
    params[p_(prefix, 'Wc_att')] = normal_weight(dim_ctx)

    # attention: hidden bias
    params[p_(prefix, 'b_att')] = np.zeros((dim_ctx,), dtype=fX)

    # attention:
    params[p_(prefix, 'U_att')] = normal_weight(dim_ctx, 1)
    params[p_(prefix, 'c_tt')] = np.zeros((1,), dtype=fX)

    return params


Layers = {
    'ff': (feed_forward, init_feed_forward),
    'gru': (gru, init_gru),
    'gru_cond': (gru_cond, init_gru_cond),
}


def get_layer(name):
    return Layers[name][0]


def get_init(name):
    return Layers[name][1]
