#! /usr/bin/python

from __future__ import print_function, unicode_literals

import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm

from config import Config, ParamConfig
from utils import load_cifar10_data, logging


# ##################### Build the neural network model #######################
@logging
def build_cnn(input_var=None, n=ParamConfig['n']):
    # create a residual learning building block with two stacked 3x3 conv-layers as in paper
    def residual_block(layer, increase_dim=False, projection=False):
        input_num_filters = layer.output_shape[1]
        if increase_dim:
            first_stride = (2, 2)
            out_num_filters = input_num_filters * 2
        else:
            first_stride = (1, 1)
            out_num_filters = input_num_filters

        stack_1 = batch_norm(
                ConvLayer(layer, num_filters=out_num_filters, filter_size=(3, 3), stride=first_stride,
                          nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_2 = batch_norm(
                ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3, 3), stride=(1, 1), nonlinearity=None,
                          pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(
                        ConvLayer(layer, num_filters=out_num_filters, filter_size=(1, 1), stride=(2, 2),
                                  nonlinearity=None, pad='same', b=None, flip_filters=False))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]), nonlinearity=rectify)
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(layer, lambda X: X[:, :, ::2, ::2],
                                           lambda s: (s[0], s[1], s[2] // 2, s[3] // 2))
                padding = PadLayer(identity, [out_num_filters // 4, 0, 0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]), nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, layer]), nonlinearity=rectify)

        return block

    # Building the network
    layer_in = InputLayer(shape=(None, 3, 32, 32), input_var=input_var)

    # first layer, output is 16 x 32 x 32
    layer = batch_norm(ConvLayer(layer_in, num_filters=16, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify,
                                 pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

    # first stack of residual blocks, output is 16 x 32 x 32
    for _ in range(n):
        layer = residual_block(layer)

    # second stack of residual blocks, output is 32 x 16 x 16
    layer = residual_block(layer, increase_dim=True)
    for _ in range(1, n):
        layer = residual_block(layer)

    # third stack of residual blocks, output is 64 x 8 x 8
    layer = residual_block(layer, increase_dim=True)
    for _ in range(1, n):
        layer = residual_block(layer)

    # average pooling
    layer = GlobalPoolLayer(layer)

    # fully connected layer
    network = DenseLayer(
            layer, num_units=10,
            W=lasagne.init.HeNormal(),
            nonlinearity=softmax)

    return network


def main():
    pass


if __name__ == '__main__':
    main()
