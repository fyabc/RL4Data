#! /usr/bin/python

from __future__ import print_function, unicode_literals

from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from config import Config, CifarConfig, PolicyConfig
from utils import fX, floatX, init_norm, logging, message
from optimizers import adadelta, adam, sgd, rmsprop

__author__ = 'fyabc'


class PolicyNetworkBase(object):
    """The base class of the policy network.

    Input the softmax probabilities of output layer of NN, output the data selection result.

    Parameters:
        input_size: the input size of the policy, should be the size of softmax probabilities of CNN.
    """

    @logging
    def __init__(self,
                 input_size,
                 optimizer=None,
                 rb_update_rate=None,
                 learning_rate=None,
                 gamma=None):
        # Load hyperparameters
        self.input_size = input_size
        self.random_generator = RandomStreams(Config['seed'])
        self.optimizer = optimizer or PolicyConfig['policy_optimizer']
        self.rb_update_rate = rb_update_rate or PolicyConfig['reward_baseline_update_rate']
        self.gamma = gamma or PolicyConfig['gamma']

        learning_rate = learning_rate or PolicyConfig['policy_learning_rate']
        self.learning_rate = theano.shared(floatX(learning_rate), name='learning_rate')

        # A batch of input softmax probabilities
        self.batch_input = T.matrix(name='batch_input', dtype=fX)

        # replay buffers
        # action_buffer is a list of {a list of actions per minibatch} per epoch
        # input_buffer is like action_buffer
        # reward_buffer is like action_buffer
        self.input_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

        # build cost and update functions
        self.reward_baseline = 0.0

        # parameters, to be filled by subclasses
        self.parameters = None

    def make_output(self, input_):
        pass

    def build_output_function(self):
        # Build output function
        self.batch_output = self.make_output()
        self.batch_output.name = 'batch_output'

        self.batch_output_sample = self.random_generator.binomial(size=self.batch_output.shape, p=self.batch_output)
        self.batch_output_sample.name = 'batch_output_sample'

        self.f_batch_output = theano.function(
            inputs=[self.batch_input],
            outputs=self.batch_output,
        )

        self.f_batch_output_sample = theano.function(
            inputs=[self.batch_input],
            outputs=self.batch_output_sample,
        )

    def build_update_function(self):
        # inputs, outputs, actions and rewards of the whole epoch
        batch_action = T.ivector('actions')
        batch_reward = T.vector('rewards', dtype=fX)

        cost = -T.mean(batch_reward * (batch_action * T.log(self.batch_output) +
                                       (1.0 - batch_action) * T.log(1.0 - self.batch_output)))

        grads = T.grad(
            cost,
            self.parameters,
        )

        # optimizers from nmt
        lr = T.scalar('lr', dtype=fX)

        param_dict = OrderedDict()
        for parameter in self.parameters:
            param_dict[parameter.name] = parameter

        self.f_grad_shared, self.f_update = eval(self.optimizer)(
            lr, param_dict, grads, [self.batch_input, batch_action, batch_reward], cost)

    def take_action(self, inputs, log_replay=True):
        actions = self.f_batch_output_sample(inputs).astype(bool)

        if log_replay:
            self.input_buffer[-1].append(inputs)
            self.action_buffer[-1].append(actions)

        return actions

    def get_discounted_rewards(self):
        # Shape of input buffer / action buffer is (epoch_num, batch_num)

        # get discounted reward
        discounted_rewards = [None] * len(self.action_buffer)

        temp = 0.
        for epoch_num, epoch_reward in reversed(list(enumerate(self.reward_buffer))):
            temp = temp * self.gamma + epoch_reward
            discounted_rewards[epoch_num] = temp

        return discounted_rewards

    def update_raw(self, inputs, actions, rewards):
        cost = self.f_grad_shared(inputs, actions, rewards)
        self.f_update(self.learning_rate.get_value())

        return cost

    def update(self, final_reward):
        cost = 0.0

        if PolicyConfig['immediate_reward']:
            discounted_rewards = self.get_discounted_rewards()

            for epoch_inputs, epoch_actions, epoch_reward in \
                    zip(self.input_buffer, self.action_buffer, discounted_rewards):
                for batch_inputs, batch_actions in zip(epoch_inputs, epoch_actions):
                    cost += self.update_raw(batch_inputs, batch_actions,
                                            np.full(batch_actions.shape, epoch_reward, dtype=fX))
        else:
            temp = final_reward - self.reward_baseline
            for epoch_inputs, epoch_actions in zip(self.input_buffer, self.action_buffer):
                for batch_inputs, batch_actions in zip(epoch_inputs, epoch_actions):
                    cost += self.update_raw(batch_inputs, batch_actions,
                                            np.full(batch_actions.shape, temp, dtype=fX))

        # clear buffers
        self.clear_buffer()

        # This may useless?
        if PolicyConfig['reward_baseline']:
            self.update_rb(final_reward)

        message("""\
Cost: {}
Real cost (Final reward for terminal): {}""".format(cost, final_reward))

        self.message_parameters()

    def start_new_epoch(self):
        self.input_buffer.append([])
        self.action_buffer.append([])

    def clear_buffer(self):
        self.input_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

    def update_rb(self, reward):
        """update reward baseline"""
        self.reward_baseline = (1 - self.rb_update_rate) * self.reward_baseline + self.rb_update_rate * reward

    @logging
    def save_policy(self, filename=None):
        filename = filename or Config['policy_model_file']
        filename = filename.replace('.npz', '_{}.npz'.format(self.input_size))
        np.savez(filename, *[parameter.get_value() for parameter in self.parameters])

    @logging
    def load_policy(self, filename=None):
        filename = filename or Config['policy_model_file']
        filename = filename.replace('.npz', '_{}.npz'.format(self.input_size))

        with np.load(filename) as f:
            for i, parameter in enumerate(self.parameters):
                parameter.set_value(f['arr_{}'.format(i)])

    def message_parameters(self):
        message('Parameters:')
        for parameter in self.parameters:
            message('$    {} = {}'.format(parameter.name, parameter.get_value()))


class LRPolicyNetwork(PolicyNetworkBase):
    def __init__(self,
                 input_size,
                 optimizer=None,
                 learning_rate=None,
                 gamma=None,
                 rb_update_rate=None,
                 start_b=None,
                 ):
        super(LRPolicyNetwork, self).__init__(input_size, optimizer, rb_update_rate, learning_rate, gamma)

        # Parameters to be learned
        self.W = theano.shared(name='W', value=init_norm(input_size))

        if start_b is None:
            start_b = PolicyConfig['b_init']
        self.b = theano.shared(name='b', value=floatX(start_b))
        self.parameters = [self.W, self.b]

        self.build_output_function()
        self.build_update_function()

    def make_output(self, input_=None):
        input_ = input_ or self.batch_input
        return T.nnet.sigmoid(T.dot(input_, self.W) + self.b)


class MLPPolicyNetwork(PolicyNetworkBase):
    def __init__(self,
                 input_size,
                 hidden_size=None,
                 optimizer=None,
                 learning_rate=None,
                 gamma=None,
                 rb_update_rate=None,
                 start_b=None):
        super(MLPPolicyNetwork, self).__init__(input_size, optimizer, rb_update_rate, learning_rate, gamma)

        self.hidden_size = hidden_size or PolicyConfig['hidden_size']

        self.W0 = theano.shared(name='W0', value=init_norm(input_size, hidden_size))
        self.b0 = theano.shared(name='b0', value=np.zeros((hidden_size,)))
        self.W1 = theano.shared(name='W1', value=init_norm(hidden_size))

        if start_b is None:
            start_b = PolicyConfig['b_init']
        self.b1 = theano.shared(name='b1', value=floatX(start_b))

        self.parameters = [self.W0, self.b0, self.W1, self.b1]

        self.build_output_function()
        self.build_update_function()

    def make_output(self, input_=None):
        input_ = input_ or self.batch_input

        hidden_layer = T.tanh(T.dot(input_, self.W0) + self.b0)

        return T.nnet.sigmoid(T.dot(hidden_layer, self.W1) + self.b1)


class LRPolicyNetwork2(object):
    """
    The simplest policy network.
    Just use logistic regression.
    """

    @logging
    def __init__(self,
                 input_size,
                 optimizer=None,
                 learning_rate=None,
                 gamma=None,
                 rb_update_rate=None,
                 start_b=None,
                 ):
        self.input_size = input_size

        # Load hyperparameters
        learning_rate = learning_rate or PolicyConfig['policy_learning_rate']
        self.learning_rate = theano.shared(floatX(learning_rate), name='learning_rate')

        self.optimizer = optimizer or PolicyConfig['policy_optimizer']
        self.gamma = gamma or PolicyConfig['gamma']
        self.rb_update_rate = rb_update_rate or PolicyConfig['reward_baseline_update_rate']
        self.random_generator = RandomStreams(Config['seed'])

        # Parameters to be learned
        self.W = theano.shared(name='W', value=init_norm(input_size))

        if start_b is None:
            start_b = PolicyConfig['b_init']
        self.b = theano.shared(name='b', value=floatX(start_b))
        self.parameters = [self.W, self.b]

        # A single case of input softmax probabilities
        self.input = T.vector(name='softmax_probabilities', dtype=fX)

        # Build output function
        self.output = self.make_output()
        self.output_sample = self.random_generator.binomial(size=self.output.shape, p=self.output)

        self.output_function = theano.function(
            inputs=[self.input],
            outputs=self.output,
        )

        self.output_sample_function = theano.function(
            inputs=[self.input],
            outputs=self.output_sample,
        )

        # replay buffers
        # action_buffer is a list of {a list of actions per minibatch} per epoch
        # input_buffer is like action_buffer
        # reward_buffer is like action_buffer
        self.input_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

        # build cost and update functions
        self.reward_baseline = 0.

        # inputs, outputs, actions and rewards of the whole epoch
        inputs = T.matrix('inputs', dtype=fX)
        outputs = self.make_output(inputs)
        actions = T.ivector('actions')
        rewards = T.vector('rewards', dtype=fX)

        cost = -T.mean(rewards * (actions * T.log(outputs) + (1.0 - actions) * T.log(1.0 - outputs)))

        grads = T.grad(
            cost,
            self.parameters,
            # known_grads={
            #     self.output: (self.output_sample - self.output)
            # }
        )

        # optimizers from nmt
        lr = T.scalar('lr', dtype=fX)

        param_dict = OrderedDict()
        param_dict['W'] = self.W
        param_dict['b'] = self.b

        self.f_grad_shared, self.f_update = eval(self.optimizer)(lr, param_dict, grads, [inputs, actions, rewards], cost)

    def make_output(self, input_=None):
        input_ = input_ or self.input
        return T.nnet.sigmoid(T.dot(input_, self.W) + self.b)

    def update_rb(self, reward):
        """update reward baseline"""
        self.reward_baseline = (1 - self.rb_update_rate) * self.reward_baseline + self.rb_update_rate * reward

    def take_action(self, inputs, log_replay=True):
        actions = np.zeros(shape=(inputs.shape[0],), dtype=bool)

        for i, input_ in enumerate(inputs):
            action = self.output_sample_function(input_)
            action = bool(action)
            actions[i] = action

        if log_replay:
            self.input_buffer[-1].append(inputs)
            self.action_buffer[-1].append(actions)

        return actions

    def discount_learning_rate(self, discount=PolicyConfig['policy_learning_rate_discount']):
        self.learning_rate.set_value(self.learning_rate.get_value() * floatX(discount))
        message('New learning rate:', self.learning_rate.get_value())

    def get_discounted_rewards(self):
        # Shape of input buffer / action buffer is (epoch_num, batch_num)

        # get discounted reward
        discounted_rewards = [None] * len(self.action_buffer)

        temp = 0.
        for epoch_num, epoch_reward in reversed(list(enumerate(self.reward_buffer))):
            temp = temp * self.gamma + epoch_reward
            discounted_rewards[epoch_num] = temp

        return discounted_rewards

    def start_new_epoch(self):
        self.input_buffer.append([])
        self.action_buffer.append([])

    def clear_buffer(self):
        self.input_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

    @logging
    def update(self, final_reward):
        cost = 0.

        if PolicyConfig['immediate_reward']:
            discounted_rewards = self.get_discounted_rewards()

            for epoch_inputs, epoch_actions, epoch_reward in \
                    zip(self.input_buffer, self.action_buffer, discounted_rewards):
                for batch_inputs, batch_actions in zip(epoch_inputs, epoch_actions):
                    cost += self.f_grad_shared(batch_inputs, batch_actions,
                                               np.full(batch_actions.shape, epoch_reward, dtype=fX))
                    self.f_update(self.learning_rate.get_value())
        else:
            temp = final_reward - self.reward_baseline
            for epoch_inputs, epoch_actions in zip(self.input_buffer, self.action_buffer):
                for batch_inputs, batch_actions in zip(epoch_inputs, epoch_actions):
                    cost += self.f_grad_shared(batch_inputs, batch_actions,
                                               np.full(batch_actions.shape, temp, dtype=fX))
                    self.f_update(self.learning_rate.get_value())

        # clear buffers
        self.clear_buffer()

        # This may useless?
        if PolicyConfig['reward_baseline']:
            self.update_rb(final_reward)

        message('Cost: {}\n'
                'Real cost (Final reward for terminal): {}\n'
                'New parameters:\n'
                '$    w = {}\n'
                '$    b = {}'
                .format(cost, final_reward, self.W.get_value(), self.b.get_value()))

    @logging
    def save_policy(self, filename=None):
        filename = filename or Config['policy_model_file']
        filename = filename.replace('.npz', '_{}.npz'.format(self.input_size))
        np.savez(filename, self.W.get_value(), self.b.get_value())

    @logging
    def load_policy(self, filename=None):
        filename = filename or Config['policy_model_file']
        filename = filename.replace('.npz', '_{}.npz'.format(self.input_size))

        with np.load(filename) as f:
            assert self.W.get_value().shape == f['arr_0'].shape,\
                'The shape of the policy W {} != the saved model shape W {}'.\
                format(self.W.get_value().shape, f['arr_0'].shape)
            self.W.set_value(f['arr_0'])
            self.b.set_value(f['arr_1'])

    def update_raw(self, inputs, actions, rewards):
        cost = self.f_grad_shared(inputs, actions, rewards)
        self.f_update(self.learning_rate.get_value())

        return cost


def test():
    pn = LRPolicyNetwork2(input_size=CifarConfig['cnn_output_size'])

    input_data = np.ones(shape=(4, CifarConfig['cnn_output_size']), dtype=fX)

    print(pn.take_action(input_data, False))


if __name__ == '__main__':
    test()
