#! /usr/bin/python

from __future__ import print_function

import os
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from .utility.config import Config, PolicyConfig
from .utility.my_logging import message, logging
from .utility.name_register import NameRegister
from .utility.utils import fX, floatX, init_norm
from .utility.optimizers import get_optimizer


class PolicyNetworkBase(NameRegister):
    """The base class of the policy network.

    Input the softmax probabilities of output layer of NN, output the data selection result.

    Parameters:
        input_size: the input size of the policy, should be the size of softmax probabilities of CNN.
    """

    NameTable = {}

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

        # build cost and update functions
        self.reward_baseline = 0.0

        # parameters, to be filled by subclasses
        self.parameters = None

    def make_output(self, input_=None):
        """Build output from input.

        Parameters
        ----------
        input_: Tensor, (batch_size, policy_input_size), optional
            batch input of features
            if not given, will use self.batch_input.

        Returns
        -------
        Tensor, (batch_size,)
            batch output of 0~1 probability values
        """

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

        cost = T.mean(batch_reward * T.nnet.binary_crossentropy(output=self.batch_output, target=batch_action))

        # Add L2 regularization
        if PolicyConfig['l2_c'] > 0.:
            l2_penalty = 0.
            for parameter in self.parameters:
                l2_penalty += (parameter ** 2).sum()
            cost += l2_penalty * PolicyConfig['l2_c']

        grads = T.grad(
            cost,
            self.parameters,
        )

        # optimizers from nmt
        lr = T.scalar('lr', dtype=fX)

        param_dict = OrderedDict()
        for parameter in self.parameters:
            param_dict[parameter.name] = parameter

        self.f_grad_shared, self.f_update = get_optimizer(
            self.optimizer, lr, param_dict, grads, [self.batch_input, batch_action, batch_reward], cost)

    def take_action(self, inputs, log_replay=True):
        actions = self.f_batch_output_sample(inputs).astype(bool)

        if log_replay:
            self.input_buffer[-1].append(inputs)
            self.action_buffer[-1].append(actions)

        return actions

    def get_discounted_rewards(self, immediate_reward):
        # Shape of input buffer / action buffer is (validation_point_num, batch_num)

        # [NOTE] Important! insert rewards into validation points
        # How to:
        #     the immediate reward of one validation part (M = 125 batches) is [r].
        #     for each batch in this part [i], get a random [r'_i] from uniform(-r, r).
        #     for each [i] in [1 ~ M-1], get [r_i] = [r'_i] - [r'_(i-1)] ([r_1] = [r'_1])
        #     so the sum of [r_i] is [r], we get the immediate reward for each batch.

        # Return value: [R_t] = [r_t] + [gamma]^1 * [r_(t+1)] + [gamma]^2 * [r_(t+2)] + ... +
        #                       [gamma]^(T-t) * [r_T]
        # Calculate return value: [R_t] = [R_(t+1)] * [gamma] + [r_t]

        # get discounted reward
        discounted_rewards = []

        for i, reward in enumerate(immediate_reward):
            part_size = len(self.action_buffer[i])
            discounted_rewards.append(np.random.uniform(-abs(reward), abs(reward), (part_size,)).astype(fX))

            dri = discounted_rewards[-1]
            dri[-1] = reward

            for j in range(part_size - 1, 0, -1):
                dri[j] -= dri[j - 1]

        temp = 0.
        for discounted_reward in reversed(discounted_rewards):
            for i in range(len(discounted_reward) - 1, -1, -1):
                temp = temp * self.gamma + discounted_reward[i]
                discounted_reward[i] = floatX(temp)

        return discounted_rewards

    def update_raw(self, inputs, actions, rewards):
        cost = self.f_grad_shared(inputs, actions, rewards)
        self.f_update(self.learning_rate.get_value())

        return cost

    @logging
    def update(self, reward_checker):
        cost = 0.0

        final_reward = reward_checker.get_reward(echo=True)

        old_parameters = [param.get_value() for param in self.parameters]

        if reward_checker.ImmediateReward:
            discounted_rewards = self.get_discounted_rewards(reward_checker.get_immediate_reward(echo=True))

            for part_inputs, part_actions, part_reward in \
                    zip(self.input_buffer, self.action_buffer, discounted_rewards):
                for batch_inputs, batch_actions, batch_reward in zip(part_inputs, part_actions, part_reward):
                    cost += self.update_raw(batch_inputs, batch_actions,
                                            np.full(batch_actions.shape, batch_reward, dtype=fX))
                if np.isnan(cost) or np.isinf(cost):
                    raise OverflowError('NaN detected at policy update')

            message('''\
ActionPartSize {} ImmediateRewardSize {}'''.format(
                len(self.action_buffer), len(discounted_rewards)
            ))
        else:
            temp = final_reward - self.reward_baseline
            for part_inputs, part_actions in reversed(zip(self.input_buffer, self.action_buffer)):
                for batch_inputs, batch_actions in zip(part_inputs, part_actions):
                    cost += self.update_raw(batch_inputs, batch_actions,
                                            np.full(batch_actions.shape, temp, dtype=fX))
                if np.isnan(cost) or np.isinf(cost):
                    raise OverflowError('NaN detected at policy update')

                # Add reward discount
                if Config['temp_job'] == 'discount_reward':
                    temp *= self.gamma

            # Reward baseline (only for terminal reward)
            if PolicyConfig['reward_baseline']:
                self.update_rb(final_reward)

        # If it is speed reward, use smooth update to reduce the speed of policy update.
        # [NOTE] ONLY for speed reward!
        if PolicyConfig['reward_checker'] == 'speed':
            smooth = floatX(PolicyConfig['smooth_update'])
            for i, param in enumerate(self.parameters):
                param.set_value(floatX(smooth * old_parameters[i] + (1 - smooth) * param.get_value()))

        message("""\
Cost: {}
Real cost (Final reward for terminal): {}""".format(
            cost, final_reward
        ))

        # clear buffers
        self.clear_buffer()

        # self.message_parameters()

    @logging
    def discount_learning_rate(self, discount_rate=0.5, linear_drop=None):
        if linear_drop is not None:
            self.learning_rate.set_value(floatX(self.learning_rate.get_value() - linear_drop))
        else:
            self.learning_rate.set_value(floatX(self.learning_rate.get_value() * discount_rate))

    def start_new_validation_point(self):
        self.input_buffer.append([])
        self.action_buffer.append([])

    start_new_epoch = start_new_validation_point

    def start_new_episode(self, episode):
        self.message_parameters()
        self.start_new_validation_point()

        d_freq = PolicyConfig['policy_learning_rate_discount_freq']
        if d_freq > 0 and (episode + 1) % d_freq == 0:
            self.discount_learning_rate(discount_rate=PolicyConfig['policy_learning_rate_discount'])

    def clear_buffer(self):
        self.input_buffer = []
        self.action_buffer = []

    def update_rb(self, reward):
        """update reward baseline"""
        self.reward_baseline = (1 - self.rb_update_rate) * self.reward_baseline + self.rb_update_rate * reward

    @logging
    def save_policy(self, filename=None, episode=0):
        filename = filename or PolicyConfig['policy_save_file']
        # filename = filename.replace('.npz', '_{}.npz'.format(self.input_size))
        root, ext = os.path.splitext(filename)
        np.savez(str('{}.{}{}'.format(root, episode, ext)), *[parameter.get_value() for parameter in self.parameters])

    @logging
    def load_policy(self, filename=None):
        filename = filename or PolicyConfig['policy_load_file']

        if not os.path.exists(filename):
            message('Policy file "{}" not exist, do not load'.format(filename))
            return

        with np.load(filename) as f:
            for i, parameter in enumerate(self.parameters):
                parameter.set_value(f['arr_{}'.format(i)])

    def message_parameters(self):
        message('Parameters:')
        for parameter in self.parameters:
            value = parameter.get_value()
            if value.ndim == 1:
                value_str = ' '.join(str(e) for e in value)
            else:
                value_str = str(value)
            message('$    {} = {}'.format(parameter.name, value_str))

    def check_load(self):
        train_action = Config['action'].lower()

        if train_action == 'reload' and PolicyConfig['start_episode'] >= 0:
            self.load_policy()


class LRPolicyNetwork(PolicyNetworkBase):
    def __init__(self,
                 input_size,
                 optimizer=None,
                 learning_rate=None,
                 gamma=None,
                 rb_update_rate=None,
                 start_b=None,
                 start_W=None,
                 ):
        super(LRPolicyNetwork, self).__init__(input_size, optimizer, rb_update_rate, learning_rate, gamma)

        # Parameters to be learned
        if start_W is None:
            start_W = PolicyConfig['W_init']
        start_W = init_norm(input_size, normalize=PolicyConfig['W_normalize']) \
            if start_W is None \
            else np.array(PolicyConfig['W_init'], dtype=fX)

        self.W = theano.shared(name='W', value=start_W)

        if start_b is None:
            start_b = PolicyConfig['b_init']
        self.b = theano.shared(name='b', value=floatX(start_b))
        self.parameters = [self.W, self.b]

        self.build_output_function()
        self.build_update_function()

    def make_output(self, input_=None):
        input_ = input_ or self.batch_input
        return T.nnet.sigmoid(T.dot(input_, self.W) + self.b)

LRPolicyNetwork.register_class(['lr'])


class MLPPolicyNetwork(PolicyNetworkBase):
    def __init__(self,
                 input_size,
                 hidden_size=None,
                 optimizer=None,
                 learning_rate=None,
                 gamma=None,
                 rb_update_rate=None,
                 start_b=None,
                 ):
        super(MLPPolicyNetwork, self).__init__(input_size, optimizer, rb_update_rate, learning_rate, gamma)

        self.hidden_size = hidden_size or PolicyConfig['hidden_size']

        self.W0 = theano.shared(name='W0', value=init_norm(self.input_size, self.hidden_size,
                                                           normalize=PolicyConfig['W_normalize']))
        self.b0 = theano.shared(name='b0', value=np.zeros((self.hidden_size,), dtype=fX))
        self.W1 = theano.shared(name='W1', value=init_norm(self.hidden_size, normalize=PolicyConfig['W_normalize']))

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

MLPPolicyNetwork.register_class(['mlp'])


class FixedPolicyNetWork(LRPolicyNetwork):
    # def start_new_validation_point(self):
    #     # If the current last buffer is empty, do not create new buffer.
    #     if not self.input_buffer or self.input_buffer[-1]:
    #         self.input_buffer.append([])
    #         self.action_buffer.append([])

    def __init__(self, *args, **kwargs):
        super(FixedPolicyNetWork, self).__init__(*args, **kwargs)
        self.reward_baseline = None

    def get_discounted_rewards(self, immediate_reward):
        # Merge all batches. The result reward size is the total number of instances in this episode.
        discounted_rewards = []

        for i, reward in enumerate(immediate_reward):
            part_size = sum(len(batches) for batches in self.action_buffer[i])
            if part_size == 0:
                continue
            if False:
                discounted_rewards.append(np.random.uniform(-abs(reward), abs(reward), (part_size,)).astype(fX))

                dri = discounted_rewards[-1]
                dri[-1] = reward

                for j in range(part_size - 1, 0, -1):
                    dri[j] -= dri[j - 1]
            else:
                reward_before = 0.0
                if discounted_rewards:
                    reward_before = discounted_rewards[-1][-1]
                discounted_rewards.append(np.linspace(reward_before, reward, part_size, endpoint=False, dtype=fX))

        temp = 0.
        for discounted_reward in reversed(discounted_rewards):
            for i in range(len(discounted_reward) - 1, -1, -1):
                temp = temp * self.gamma + discounted_reward[i]
                discounted_reward[i] = floatX(temp)

        discounted_rewards = np.concatenate(discounted_rewards, axis=0)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards

    @logging
    def update(self, reward_checker):
        old_parameters = [param.get_value() for param in self.parameters]

        def _concat(a):
            result = []
            for vp in a:
                for b in vp:
                    result.append(b)
            return np.concatenate(result, axis=0)

        input_buffer = _concat(self.input_buffer)
        action_buffer = _concat(self.action_buffer)
        assert input_buffer.shape[0] == action_buffer.shape[0]

        final_reward = reward_checker.get_reward(echo=True)

        if reward_checker.ImmediateReward:
            imm_reward = reward_checker.get_immediate_reward(echo=True)
            discounted_rewards = self.get_discounted_rewards(imm_reward)
        else:
            # # Calculate fake linear immediate reward
            # acc_vp_size_list = []
            # acc_vp_size = 0
            # for part in self.action_buffer:
            #     acc_vp_size += len(part)
            #     acc_vp_size_list.append(acc_vp_size)
            # acc_vp_size_list = np.array(acc_vp_size_list, dtype=fX)
            # imm_reward = acc_vp_size_list / acc_vp_size_list[-1] * final_reward

            # discounted_rewards = np.full(action_buffer.shape, self.get_terminal_reward(final_reward), dtype=fX)

            discounted_rewards = np.linspace(final_reward, 0.0, len(action_buffer), dtype=fX)
            discounted_rewards -= discounted_rewards.mean()
            discounted_rewards /= discounted_rewards.std()

        cost = self.update_raw(input_buffer, action_buffer, discounted_rewards)

        if np.isnan(cost) or np.isinf(cost):
            raise OverflowError('NaN detected at policy update')

        msg = '    Cost: {}\n    Real cost (Final reward for terminal): {}'.format(cost, final_reward)
        if reward_checker.ImmediateReward:
            msg += '\n    Average immediate reward: {}'.format(np.mean(imm_reward))
        message(msg)

        # clear buffers
        self.clear_buffer()

        new_parameters = [param.get_value() for param in self.parameters]
        print('Param delta:', [n - o for n, o in zip(new_parameters, old_parameters)])

    def get_terminal_reward(self, termial_reward):
        """Get terminal reward and update reward baseline."""

        if self.reward_baseline is None:
            self.reward_baseline = termial_reward
            return 0.0

        result = termial_reward - self.reward_baseline
        self.update_rb(termial_reward)

        return result


FixedPolicyNetWork.register_class(['fixed'])
