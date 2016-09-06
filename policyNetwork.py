#! /usr/bin/python

from __future__ import print_function, unicode_literals

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from config import Config, ParamConfig
from utils import fX, floatX, init_norm, logging, message
from optimizers import adadelta, adam, sgd, rmsprop

__author__ = 'fyabc'


class PolicyNetwork(object):

    """
    The policy network.
    Input the softmax probabilities of output layer of NN, output the data selection result.
    Just use logistic regression.

    Parameters:
        input_size: the input size of the policy, should be the size of softmax probabilities of CNN.
    """

    @logging
    def __init__(self,
                 input_size=ParamConfig['cnn_output_size'],
                 optimizer=ParamConfig['policy_optimizer'],
                 learning_rate=ParamConfig['policy_learning_rate'],
                 gamma=ParamConfig['gamma'],
                 rb_update_rate=ParamConfig['reward_baseline_update_rate']
                 ):

        theano.config.exception_verbosity = 'high'

        self.input_size = input_size
        self.learning_rate = theano.shared(floatX(learning_rate), name='learning_rate')
        self.gamma = gamma
        self.rb_update_rate = rb_update_rate
        self.random_generator = RandomStreams(Config['seed'])

        # parameters to be learned
        self.W = theano.shared(name='W', value=init_norm(input_size))
        self.b = theano.shared(name='b', value=floatX(2.))
        self.parameters = [self.W, self.b]

        # a single case of input softmax probabilities
        self.input = T.vector(name='softmax_probabilities', dtype=fX)

        # build computation graph of output
        self.output = self.make_output(self.input)
        self.output_sample = self.random_generator.binomial(size=self.output.shape, p=self.output)

        self.output_function = theano.function(
            inputs=[self.input],
            outputs=[self.output],
        )

        self.output_sample_function = theano.function(
            inputs=[self.input],
            outputs=self.output_sample,
        )

        # replay buffers
        self.input_buffer = []
        self.action_buffer = []

        # build cost and update functions
        self.reward_baseline = 0.

        # inputs, outputs, actions and rewards of the whole epoch
        inputs = T.matrix('inputs', dtype=fX)
        outputs = self.make_output(inputs)
        actions = T.ivector('actions')
        rewards = T.vector('rewards', dtype=fX)

        cost = -T.sum(rewards * (actions * T.log(outputs) + (1.0 - actions) * T.log(1.0 - outputs)))

        grads = T.grad(
            cost,
            self.parameters,
            # known_grads={
            #     self.output: (self.output_sample - self.output)
            # }
        )

        # optimizers from nmt
        lr = T.scalar('lr', dtype=fX)

        from collections import OrderedDict
        param_dict = OrderedDict()
        param_dict['W'] = self.W
        param_dict['b'] = self.b

        self.f_grad_shared, self.f_update = eval(optimizer)(lr, param_dict, grads, [inputs, actions, rewards], cost)

    def make_output(self, input_):
        return T.nnet.sigmoid(T.dot(input_, self.W) + self.b)

    def take_action(self, inputs):
        actions = np.zeros(shape=(inputs.shape[0],), dtype=bool)

        for i, input_ in enumerate(inputs):
            action = self.output_sample_function(input_)
            action = bool(action)
            actions[i] = action

            self.input_buffer.append(input_)
            self.action_buffer.append(action)
        return actions

    def discount_learning_rate(self, discount=ParamConfig['policy_learning_rate_discount']):
        self.learning_rate.set_value(self.learning_rate.get_value() * floatX(discount))
        message('New learning rate:', self.learning_rate.get_value())

    @logging
    def update(self, reward):
        # TODO preprocess of reward

        # get discounted rewards
        discounted_rewards = np.zeros_like(self.action_buffer, dtype=fX)
        temp = reward - self.reward_baseline
        for i in reversed(xrange(discounted_rewards.size)):
            discounted_rewards[i] = temp
            temp *= self.gamma

        cost = self.f_grad_shared(self.input_buffer, self.action_buffer, discounted_rewards)
        self.f_update(self.learning_rate.get_value())

        # clear buffers
        self.input_buffer = []
        self.action_buffer = []

        # update reward baseline
        self.reward_baseline = (1 - self.rb_update_rate) * self.reward_baseline + self.rb_update_rate * reward

        message('Cost: {}\n'
                'New parameters:\n'
                '$    w = {}\n'
                '$    b = {}\n'
                'New reward baseline: {}'
                .format(cost, self.W.get_value(), self.b.get_value(), self.reward_baseline))

    def update_and_validate(self, reward, validate_probability):
        # TODO preprocess of reward

        # get discounted rewards
        discounted_rewards = np.zeros_like(self.action_buffer, dtype=fX)
        temp = reward - self.reward_baseline
        for i in reversed(xrange(discounted_rewards.size)):
            discounted_rewards[i] = temp
            temp *= self.gamma

        # update parameters
        # self.update_function(self.input_buffer, self.action_buffer, discounted_rewards)

        cost = self.f_grad_shared(self.input_buffer, self.action_buffer, discounted_rewards)
        self.f_update(self.learning_rate.get_value())

        # get validation cost
        validate_actions = self.take_action(validate_probability)
        validate_cost = self.f_grad_shared(validate_probability, validate_actions,
                                           [floatX(temp)] * validate_actions.shape[0])

        # clear buffers
        self.input_buffer = []
        self.action_buffer = []

        # update reward baseline
        self.reward_baseline = (1 - self.rb_update_rate) * self.reward_baseline + self.rb_update_rate * reward

        message(
            'Cost: {}\n'
            'Raw Cost: {}\n'
            'Validation Cost: {}\n'
            'Raw Validation Cost: {}\n'
            'New parameters:\n'
            '$    w = {}\n'
            '$    b = {}\n'
            'New reward baseline: {}'
            .format(
                cost, cost / temp,
                validate_cost, validate_cost / temp,
                self.W.get_value(), self.b.get_value(),
                self.reward_baseline
            )
        )

    @logging
    def save_policy(self, filename=Config['policy_model_file']):
        filename = filename.replace('.npz', '_{}.npz'.format(self.input_size))
        np.savez(filename, self.W.get_value(), self.b.get_value())

    @logging
    def load_policy(self, filename=Config['policy_model_file']):
        filename = filename.replace('.npz', '_{}.npz'.format(self.input_size))

        with np.load(filename) as f:
            assert self.W.get_value().shape == f['arr_0'].shape,\
                'The shape of the policy W {} != the saved model shape W {}'.\
                format(self.W.get_value().shape, f['arr_0'].shape)
            self.W.set_value(f['arr_0'])
            self.b.set_value(f['arr_1'])


def test():
    pn = PolicyNetwork()

    input_data = np.ones(shape=(4, ParamConfig['cnn_output_size']), dtype=fX)

    print(pn.take_action(input_data))


if __name__ == '__main__':
    test()
