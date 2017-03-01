#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from layer_NMT import get_layer, get_init, dropout
from ..utility.utils import fX, message, floatX
from ..utility.config import Config, NMTConfig as ParamConfig
from ..utility.NMT import normal_weight, concatenate, linear, Profile
from ..utility.optimizers import sgd, adam, adadelta, rmsprop
from .model import ModelBase

__author__ = 'fyabc'


class NMTModelBase(ModelBase):
    pass


class NMTModel(NMTModelBase):
    def __init__(self, train_or_translate=True):
        """
        :param train_or_translate: If True, this model is used for train, or it is used for translate.
        """

        self.np_parameters = OrderedDict()
        self.parameters = OrderedDict()

        self.learning_rate = floatX(ParamConfig['learning_rate'])

        if train_or_translate:
            message('Building model...', end=' ')
            self.init_np_parameters()

            # todo: reload model parameters

            self.init_parameters()

            self.build_train_function()
            self.build_validate_function()
        else:
            pass

    def init_np_parameters(self):
        """Initialize numpy parameters."""

        # embedding
        self.np_parameters['Wemb'] = normal_weight(ParamConfig['n_words_src'], ParamConfig['dim_word'])
        self.np_parameters['Wemb_dec'] = normal_weight(ParamConfig['n_words'], ParamConfig['dim_word'])

        # encoder: bidirectional RNN
        get_init(ParamConfig['encoder'])(
            self.np_parameters, prefix='encoder', n_in=ParamConfig['dim_word'], dim=ParamConfig['dim'])
        get_init(ParamConfig['encoder'])(
            self.np_parameters, prefix='encoder_r', n_in=ParamConfig['dim_word'], dim=ParamConfig['dim'])

        ctx_dim = 2 * ParamConfig['dim']

        # init_state, init_cell
        get_init('ff')(self.np_parameters, prefix='ff_state', n_in=ctx_dim, n_out=ParamConfig['dim'])

        # decoder
        get_init(ParamConfig['decoder'])(self.np_parameters, prefix='decoder', n_in=ParamConfig['dim_word'],
                                         dim=ParamConfig['dim'], dim_ctx=ctx_dim)

        # readout
        get_init('ff')(self.np_parameters, prefix='ff_logit_lstm', n_in=ParamConfig['dim'],
                       n_out=ParamConfig['dim_word'], orthogonal=False)
        get_init('ff')(self.np_parameters, prefix='ff_logit_prev', n_in=ParamConfig['dim_word'],
                       n_out=ParamConfig['dim_word'], orthogonal=False)
        get_init('ff')(self.np_parameters, prefix='ff_logit_ctx', n_in=ctx_dim,
                       n_out=ParamConfig['dim_word'], orthogonal=False)
        get_init('ff')(self.np_parameters, prefix='ff_logit', n_in=ParamConfig['dim_word'],
                       n_out=ParamConfig['n_words'], orthogonal=False)

    def init_parameters(self):
        """Initialize Theano tensor parameters."""

        for name, value in self.np_parameters.iteritems():
            self.parameters[name] = theano.shared(value, name=name)

    def build_model(self):
        """Build a training model."""

        self.rand = RandomStreams(Config['seed'])
        self.use_noise = theano.shared(np.float32(0.))

        # description string: #words * #samples
        self.x = T.matrix('x', dtype='int64')
        self.x_mask = T.matrix('x_mask', dtype=fX)
        self.y = T.matrix('y', dtype='int64')
        self.y_mask = T.matrix('y_mask', dtype=fX)

        self.inputs = [self.x, self.x_mask, self.y, self.y_mask]

        # for the backward rnn, we just need to invert x and x_mask
        xr = self.x[::-1]
        xr_mask = self.x_mask[::-1]

        n_time_steps = self.x.shape[0]
        n_time_steps_trg = self.y.shape[0]
        n_samples = self.x.shape[1]

        # word embedding for forward rnn (source)
        emb = self.parameters['Wemb'][self.x.flatten()].reshape(
            [n_time_steps, n_samples, ParamConfig['dim_word']])
        proj = get_layer(ParamConfig['encoder'])(emb, self.parameters, prefix='encoder', mask=self.x_mask)

        # word embedding for backward rnn (source)
        emb_r = self.parameters['Wemb'][xr.flatten()].reshape(
            [n_time_steps, n_samples, ParamConfig['dim_word']])
        proj_r = get_layer(ParamConfig['encoder'])(emb_r, self.parameters, prefix='encoder_r', mask=xr_mask)

        # context will be the concatenation of forward and backward rnn
        ctx = concatenate([proj[0], proj_r[0][::-1]], axis=proj[0].ndim - 1)

        # mean of the context (across time) will be used to initialize decoder rnn
        self.ctx_mean = (ctx * self.x_mask[:, :, None]).sum(0) / self.x_mask.sum(0)[:, None]

        # or you can use the last state of forward + backward encoder rnn
        # self.ctx_mean = concatenate([proj[0][-1], proj_r[0][-1]], axis=proj[0].ndim - 2)

        # initial decoder state
        init_state = get_layer('ff')(self.ctx_mean, self.parameters, prefix='ff_state', activation=T.tanh)

        # word embedding (target), we will shift the target sequence one time step
        # to the right. This is done because of the bi-gram connections in the
        # readout and decoder rnn. The first target will be all zeros and we will
        # not condition on the last output.
        emb = self.parameters['Wemb_dec'][self.y.flatten()].reshape(
            [n_time_steps_trg, n_samples, ParamConfig['dim_word']])
        emb = T.set_subtensor(T.zeros_like(emb)[1:], emb[:-1])

        # decoder - pass through the decoder conditional gru with attention
        proj = get_layer(ParamConfig['decoder'])(
            emb, self.parameters, prefix='decoder', mask=self.y_mask,
            context=ctx, context_mask=self.x_mask, one_step=False, init_state=init_state)

        # hidden states of the decoder gru
        proj_h = proj[0]  # n_time_steps * n_sample * dim

        # weighted averages of context, generated by attention module
        ctx_s = proj[1]

        # weights (alignment matrix)
        self.dec_alphas = proj[2]

        # compute word probabilities
        logit_lstm = get_layer('ff')(proj_h, self.parameters, prefix='ff_logit_lstm', activation=linear)
        logit_prev = get_layer('ff')(emb, self.parameters, prefix='ff_logit_prev', activation=linear)
        logit_ctx = get_layer('ff')(ctx_s, self.parameters, prefix='ff_logit_ctx', activation=linear)

        logit = T.tanh(logit_lstm + logit_prev + logit_ctx)  # n_time_steps * n_sample * dim_word
        if ParamConfig['use_dropout']:
            logit = dropout(logit, self.use_noise, self.rand)
        # n_time_steps * n_sample * n_words
        logit = get_layer('ff')(logit, self.parameters, prefix='ff_logit', activation=linear)

        logit_shape = logit.shape
        probs = T.nnet.softmax(logit.reshape([logit_shape[0] * logit_shape[1], logit_shape[2]]))

        # cost
        y_flat = self.y.flatten()
        y_flat_index = T.arange(y_flat.shape[0]) * ParamConfig['n_words'] + y_flat

        self.cost = -T.log(probs.flatten()[y_flat_index]).reshape([self.y.shape[0], self.y.shape[1]])
        self.cost = (self.cost * self.y_mask).sum(0)

    def build_train_function(self):
        message('Building f_log_probs...', end=' ')

        self.f_log_probs = theano.function(
            self.inputs,
            self.cost,
            profile=Profile,
        )

        self.f_x_emb = theano.function(
            [self.x, self.x_mask],
            self.ctx_mean,
            profile=Profile,
        )

        message('Done')

        cost = self.cost.mean()

        # apply L2 regularization on weights
        if ParamConfig['decay_c'] > 0.:
            decay_c = theano.shared(floatX(ParamConfig['decay_c']), name='decay_c')
            weight_decay = sum(((v ** 2).sum() for v in self.parameters.itervalues()), 0.0) * decay_c

            cost += weight_decay

        # regularize the alpha weights
        if ParamConfig['alpha_c'] > 0. and not ParamConfig['decoder'].endswith('simple'):
            alpha_c = theano.shared(floatX(ParamConfig['alpha_c']), name='alpha_c')

            cost += alpha_c * (
                (T.cast(self.y_mask.sum(0) // self.x_mask.sum(0), fX)[:, None] - self.dec_alphas.sum(0)) ** 2
            ).sum(1).mean()

        # after all regularizers - compile the computational graph for cost
        message('Building f_cost...', end=' ')
        self.f_cost = theano.function(self.inputs, cost, profile=Profile)
        message('Done')

        message('Computing gradient...', end=' ')
        grads = T.grad(cost, wrt=self.parameters.values())
        message('Done')

        # apply gradient clipping here
        if ParamConfig['clip_c'] > 0.:
            clip_c = ParamConfig['clip_c']
            g2 = sum(((g ** 2).sum() for g in grads), 0.)

            grads = [T.switch(
                g2 > (clip_c ** 2),
                g / T.sqrt(g2) * clip_c,
                g
            ) for g in grads]

        # compile the optimizer, the actual computational graph is compiled here
        lr = T.scalar(name='lr')

        message('Building optimizers...', end=' ')
        self.f_grad_shared, self.f_update = eval(ParamConfig['optimizer'])(
            lr, self.parameters, grads, self.inputs, cost)
        message('Done')

    def f_train(self, x, x_mask, y, y_mask):
        if x.shape[1] == 0:
            return None

        cost = self.f_grad_shared(x, x_mask, y, y_mask)
        self.f_update(self.learning_rate)

        return cost

    def build_sampler(self):
        """Build a sampler."""

        x = T.matrix('x', dtype='int64')
        xr = x[::-1]

        n_time_steps = x.shape[0]
        n_samples = x.shape[1]

        # word embedding (source), forward and backward
        emb = self.parameters['Wemb'][x.flatten()].reshape(
            [n_time_steps, n_samples, ParamConfig['dim_word']])
        emb_r = self.parameters['Wemb'][xr.flatten()].reshape(
            [n_time_steps, n_samples, ParamConfig['dim_word']])

        # encoder
        proj = get_layer(ParamConfig['encoder'])(emb, self.parameters, prefix='encoder')
        proj_r = get_layer(ParamConfig['encoder'])(emb_r, self.parameters, prefix='encoder_r')

        # concatenate forward and backward rnn hidden states
        ctx = concatenate([proj[0], proj_r[0][::-1]], axis=proj[0].ndim - 1)

        # get the input for decoder rnn initializer mlp
        ctx_mean = ctx.mean(0)
        # ctx_mean = concatenate([proj[0][-1], proj_r[0][-1]], axis=proj[0].ndim - 2)

        init_state = get_layer('ff')(ctx_mean, self.parameters, prefix='ff_state', activation=T.tanh)

        message('Building f_init...', end=' ')
        self.f_init = theano.function([x], [init_state, ctx], name=str('f_init'), profile=Profile)
        message('Done')

        # x: 1 * 1
        y = T.vector('y_sampler', dtype='int64')
        init_state = T.matrix('init_state', dtype=fX)

        # if it's the first word, emb should be all zero and it is indicated by -1
        emb = T.switch(
            y[:, None] < 0,
            T.alloc(0., 1, self.parameters['Wemb_dec'].shape[1]),
            self.parameters['Wemb_dec'][y],
        )

        # apply one step of conditional gru with attention
        proj = get_layer(ParamConfig['decoder'])(emb, self.parameters, prefix='decoder', mask=None, context=ctx,
                                                 one_step=True, init_state=init_state)

        # get the next hidden state
        next_state = proj[0]

        # get the weighted averages of context for this target word y
        ctx_s = proj[1]

        logit_lstm = get_layer('ff')(next_state, self.parameters, prefix='ff_logit_lstm', activation=linear)
        logit_prev = get_layer('ff')(emb, self.parameters, prefix='ff_logit_prev', activation=linear)
        logit_ctx = get_layer('ff')(ctx_s, self.parameters, prefix='ff_logit_ctx', activation=linear)

        logit = T.tanh(logit_lstm + logit_prev + logit_ctx)
        if ParamConfig['use_dropout']:
            logit = dropout(logit, self.use_noise, self.rand)
        logit = get_layer('ff')(logit, self.parameters, prefix='ff_logit', activation=linear)

        # compute the softmax probability
        next_probs = T.nnet.softmax(logit)

        # sample from softmax distribution to get the sample
        next_sample = self.rand.multinomial(pvals=next_probs).argmax(1)

        # compile a function to do the whole thing above, next word probability,
        # sampled word for the next target, next hidden state to be used
        message('Building f_next...', end=' ')
        self.f_next = theano.function([y, ctx, init_state], [next_probs, next_sample, next_state],
                                      name=str('f_next'), profile=Profile)
        message('Done')

    def gen_sample(self, x, rand=None, k=1, maxlen=30, stochastic=True, argmax=False):
        """Generate sample, either with stochastic sampling or beam search. Note that,
        this function iteratively calls f_init and f_next functions.
        """


def just_ref():
    # Just ref them, or they may be optimized out by PyCharm.
    _ = sgd, adam, adadelta, rmsprop
