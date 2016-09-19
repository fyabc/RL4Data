#! /usr/bin/python

from __future__ import print_function, unicode_literals

import time
import numpy as np
import cPickle as pkl
from collections import OrderedDict

import theano.tensor as T
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import fX, floatX, logging, get_minibatches_idx, message
from utils_IMDB import prepare_imdb_data as prepare_data, pr, ortho_weight
from optimizers import adadelta, adam, sgd, rmsprop
from config import IMDBConfig

__author__ = 'fyabc'


class IMDBModel(object):
    def __init__(self, reload_model=False):
        self.train_batch_size = IMDBConfig['train_batch_size']
        self.validate_batch_size = IMDBConfig['validate_batch_size']
        self.learning_rate = floatX(IMDBConfig['learning_rate'])

        # Parameters of the model (Theano shared variables)
        self.parameters = OrderedDict()

        # Build train function and parameters
        self.use_noise = None
        self.inputs = None
        self.mask = None
        self.targets = None
        self.cost = None

        self.f_cost = None
        self.f_grad = None
        self.f_predict = None
        self.f_predict_prob = None
        self.f_grad_shared = None
        self.f_update = None

        self.build_train_function()

        if reload_model:
            self.load_model()

    def init_parameters(self):
        np_parameters = OrderedDict()

        rands = np.random.rand(IMDBConfig['n_words'], IMDBConfig['dim_proj'])

        # embedding
        np_parameters['Wemb'] = (0.01 * rands).astype(fX)

        # LSTM
        self.init_lstm_parameters(np_parameters)

        # classifier
        np_parameters['U'] = 0.01 * np.random.randn(IMDBConfig['dim_proj'], IMDBConfig['ydim']).astype(fX)
        np_parameters['b'] = np.zeros((IMDBConfig['ydim'],)).astype(fX)

        # numpy parameters to shared variables
        for key, value in np_parameters.iteritems():
            self.parameters[key] = theano.shared(value, name=key)

    @staticmethod
    def init_lstm_parameters(np_parameters):
        prefix = 'lstm'
        W = np.concatenate([ortho_weight(IMDBConfig['dim_proj']),
                            ortho_weight(IMDBConfig['dim_proj']),
                            ortho_weight(IMDBConfig['dim_proj']),
                            ortho_weight(IMDBConfig['dim_proj'])], axis=1)
        np_parameters[pr(prefix, 'W')] = W
        U = np.concatenate([ortho_weight(IMDBConfig['dim_proj']),
                            ortho_weight(IMDBConfig['dim_proj']),
                            ortho_weight(IMDBConfig['dim_proj']),
                            ortho_weight(IMDBConfig['dim_proj'])], axis=1)
        np_parameters[pr(prefix, 'U')] = U
        b = np.zeros((4 * IMDBConfig['dim_proj'],))
        np_parameters[pr(prefix, 'b')] = b.astype(fX)

    @staticmethod
    def lstm_layer(tparams, state_below, mask=None):
        prefix = 'lstm'

        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1

        assert mask is not None

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def _step(m_, x_, h_, c_):
            preact = T.dot(h_, tparams[pr(prefix, 'U')])
            preact += x_

            i = T.nnet.sigmoid(_slice(preact, 0, IMDBConfig['dim_proj']))
            f = T.nnet.sigmoid(_slice(preact, 1, IMDBConfig['dim_proj']))
            o = T.nnet.sigmoid(_slice(preact, 2, IMDBConfig['dim_proj']))
            c = T.tanh(_slice(preact, 3, IMDBConfig['dim_proj']))

            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * T.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h, c

        state_below = (T.dot(state_below, tparams[pr(prefix, 'W')]) +
                       tparams[pr(prefix, 'b')])

        dim_proj = IMDBConfig['dim_proj']
        rval, updates = theano.scan(
                _step,
                sequences=[mask, state_below],
                outputs_info=[T.alloc(floatX(0.),
                                      n_samples,
                                      dim_proj),
                              T.alloc(floatX(0.),
                                      n_samples,
                                      dim_proj)],
                name=pr(prefix, '_layers'),
                n_steps=nsteps
        )

        return rval[0]

    @staticmethod
    def dropout_layer(state_before, use_noise, trng):
        proj = T.switch(
            use_noise,
            (state_before *
             trng.binomial(state_before.shape,
                           p=0.5, n=1,
                           dtype=state_before.dtype)),
            state_before * 0.5
        )
        return proj

    @logging
    def build_train_function(self):
        theano.config.exception_verbosity = 'high'
        theano.config.optimizer = 'None'

        # Initialize self.parameters
        self.init_parameters()

        trng = RandomStreams(IMDBConfig['seed'])

        # Build Theano tensor variables.

        # Used for dropout.
        self.use_noise = theano.shared(floatX(0.))

        self.inputs = T.matrix('inputs', dtype='int64')
        self.mask = T.matrix('mask', dtype=fX)
        self.targets = T.vector('targets', dtype='int64')

        n_timesteps = self.inputs.shape[0]
        n_samples = self.inputs.shape[1]

        emb = self.parameters['Wemb'][self.inputs.flatten()].reshape([n_timesteps, n_samples, IMDBConfig['dim_proj']])

        proj = self.lstm_layer(self.parameters, emb, self.mask)

        proj = (proj * self.mask[:, :, None]).sum(axis=0)
        proj = proj / self.mask.sum(axis=0)[:, None]

        if IMDBConfig['use_dropout']:
            proj = self.dropout_layer(proj, self.use_noise, trng)
            
        pred = T.nnet.softmax(T.dot(proj, self.parameters['U']) + self.parameters['b'])

        self.f_predict_prob = theano.function([self.inputs, self.mask], pred, name='f_pred_prob')
        self.f_predict = theano.function([self.inputs, self.mask], pred.argmax(axis=1), name='f_pred')
    
        off = 1e-8
        if pred.dtype == 'float16':
            off = 1e-6
    
        self.cost = -T.log(pred[T.arange(n_samples), self.targets] + off).mean()

        # return self.use_noise, self.inputs, self.mask, self.targets, self.f_predict_prob, self.f_predict, self.cost

        decay_c = IMDBConfig['decay_c']
        if decay_c > 0.:
            decay_c = theano.shared(floatX(decay_c), name='decay_c')
            weight_decay = 0.
            weight_decay += (self.parameters['U'] ** 2).sum()
            weight_decay *= decay_c
            self.cost += weight_decay

        self.f_cost = theano.function([self.inputs, self.mask, self.targets], self.cost, name='f_cost')

        grads = T.grad(self.cost, wrt=list(self.parameters.values()))
        self.f_grad = theano.function([self.inputs, self.mask, self.targets], grads, name='f_grad')

        lr = T.scalar('lr', dtype=fX)
        self.f_grad_shared, self.f_update = eval(IMDBConfig['optimizer'])(
            lr, self.parameters, grads, [self.inputs, self.mask, self.targets], self.cost)

    @logging
    def load_model(self, filename=None):
        filename = filename or IMDBConfig['save_to']

    @logging
    def save_model(self, filename=None):
        filename = filename or IMDBConfig['save_to']

    def predict_error(self, data_x, data_y, batch_indices, verbose=False):
        """
        Just compute the error
        f_pred: Theano fct computing the prediction
        prepare_data: usual prepare_data for that dataset.
        """

        valid_err = 0
        for _, valid_index in batch_indices:
            x, mask, y = prepare_data([data_x[t] for t in valid_index],
                                      np.array(data_y)[valid_index],
                                      maxlen=None)
            predicts = self.f_predict(x, mask)
            targets = np.array(data_y)[valid_index]
            valid_err += (predicts == targets).sum()
        valid_err = 1. - floatX(valid_err) / len(data_x)

        return valid_err

    def get_parameter_values(self):
        result = OrderedDict()
        for key in self.parameters:
            result[key] = self.parameters[key].get_value()
        return result

    @logging
    def train(self, train_x, train_y, valid_x, valid_y, test_x, test_y):
        history_errs = []
        best_parameters = {}
        bad_counter = 0

        train_size = len(train_x)
        valid_size = len(valid_x)
        test_size = len(test_x)

        kf_valid = get_minibatches_idx(valid_size, self.validate_batch_size)
        kf_test = get_minibatches_idx(test_size, self.validate_batch_size)

        valid_freq = IMDBConfig['valid_freq']
        if valid_freq == -1:
            valid_freq = train_size // self.train_batch_size

        save_freq = IMDBConfig['save_freq']
        if save_freq == -1:
            save_freq = train_size // self.train_batch_size

        display_freq = IMDBConfig['display_freq']
        save_to = IMDBConfig['save_to']
        patience = IMDBConfig['patience']

        update_index = 0  # the number of update done
        early_stop = False  # early stop
        start_time = time.time()

        epoch = 0
        try:
            for epoch in range(IMDBConfig['max_epochs']):
                n_samples = 0

                # Get new shuffled index for the training set.
                kf = get_minibatches_idx(train_size, self.train_batch_size, shuffle=True)

                for _, train_index in kf:
                    update_index += 1
                    self.use_noise.set_value(floatX(1.))

                    # Select the random examples for this minibatch
                    x = [train_x[t] for t in train_index]
                    y = [train_y[t] for t in train_index]

                    # Get the data in numpy.ndarray format
                    # This swap the axis!
                    # Return something of shape (minibatch maxlen, n samples)
                    x, mask, y = prepare_data(x, y)
                    n_samples += x.shape[1]

                    cost = self.f_grad_shared(x, mask, y)
                    self.f_update(self.learning_rate)

                    if np.isnan(cost) or np.isinf(cost):
                        print('bad cost detected: ', cost)
                        return 1., 1., 1.

                    if update_index % display_freq == 0:
                        print('Epoch ', epoch, 'Update ', update_index, 'Cost ', cost)

                    if save_to and update_index % save_freq == 0:
                        print('Saving...')

                        if best_parameters:
                            params = best_parameters
                        else:
                            params = self.get_parameter_values()
                        np.savez(save_to, history_errs=history_errs, **params)
                        # pkl.dump(IMDBConfig, open('%s.pkl' % save_to, 'wb'))
                        print('Done')

                    if update_index % valid_freq == 0:
                        self.use_noise.set_value(0.)
                        train_err = self.predict_error(train_x, train_y, kf)
                        valid_err = self.predict_error(valid_x, valid_y, kf_valid)
                        test_err = self.predict_error(test_x, test_y, kf_test)

                        history_errs.append([valid_err, test_err])

                        if not best_parameters or valid_err <= np.array(history_errs)[:, 0].min():
                            best_parameters = self.get_parameter_values()
                            bad_counter = 0

                        print(('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err))

                        if len(history_errs) > patience and valid_err >= np.array(history_errs)[:-patience, 0].min():
                            bad_counter += 1
                            if bad_counter > patience:
                                print('Early Stop!')
                                early_stop = True
                                break

                print('Seen %d samples' % n_samples)

                if early_stop:
                    break
        except KeyboardInterrupt:
            print('Training interrupted')

        end_time = time.time()

        kf_train_sorted = get_minibatches_idx(train_size, self.train_batch_size)

        train_err = self.predict_error(train_x, train_y, kf_train_sorted)
        valid_err = self.predict_error(valid_x, valid_y, kf_valid)
        test_err = self.predict_error(test_x, test_y, kf_test)

        print('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err)

        if IMDBConfig['save_to']:
            np.savez(IMDBConfig['save_to'], train_err=train_err,
                     valid_err=valid_err, test_err=test_err,
                     history_errs=history_errs, **best_parameters)

        print('The code run for %d epochs, with %f sec/epochs' % (
            (epoch + 1), (end_time - start_time) / (1. * (epoch + 1))))
        message(('Training took %.1fs' % (end_time - start_time)))
        return train_err, valid_err, test_err
