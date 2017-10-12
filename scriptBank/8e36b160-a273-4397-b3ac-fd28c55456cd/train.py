#!env python
# coding: utf-8
# Dep: learn for crowd2
# at - c8ad0d1

import tensorflow as tf
from lc import train, config, analysis, Loader
from tensorflow.contrib.layers import fully_connected, summarize_collection, dropout

config.NUM_UNIT = 50
config.LEARNING_RATE = 0.01
config.DECAY_STEP = 20
config.DECAY_RATE = 0.96
config.DATASIZE = 256

def max_out(inputs, num_units=config.NUM_UNIT, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs

def five_layers_lrelu(x,ref_y, test):
    test = None if not test else True
    activation = max_out
    hid1 = fully_connected(x, 1000, activation_fn=activation, reuse=test, scope="layer1")
    hid2 = fully_connected(hid1, 1000, activation_fn=activation, reuse=test, scope="layer2")
    hid3 = fully_connected(hid2, 1000, activation_fn=activation, reuse=test, scope="layer3")
    hid4 = fully_connected(hid3, 1000, activation_fn=activation, reuse=test, scope="layer4")
    hid5 = fully_connected(hid4, 1000, activation_fn=activation, reuse=test, scope="layer5")
    y = fully_connected(hid5, 1, activation_fn=tf.identity, reuse=test, scope="fc")
    if not test:
        analysis.add_RMSE_loss(y, ref_y, "train")
        analysis.add_L2_loss()
    else:
        analysis.add_RMSE_loss(y, ref_y, "test")

def apply_graph(graph):
    g1 = tf.Graph()
    with g1.as_default():
        x1, y1 = l.shuffle_batch(batch_size=config.DATASIZE)
        graph(x1, y1, False)

        x2, y2 = l.validation()
        graph(x2, y2, True)

        summarize_collection("trainable_variables")
        summarize_collection("losses")
        summarize_collection("visuals")
    return g1


# config.RESTORE_FROM = "08-06-17_21:19"

config.DATAFILE = "learn_ple.dat"
d = {
    "name": "pleLambda",
    "discription": "productive and new lc lib test"
}
l = Loader(d)

for i in range(6):
    config.L2_LAMBDA = 0.3**i
    print("\n PLE train {:.2f} lambda \n".format(config.L2_LAMBDA))
    with apply_graph(five_layers_lrelu).as_default():
        train.adaptive_train(20000)

config.DATAFILE = "learn_cls.dat"
d = {
    "name": "clsLambda",
    "discription": "productive and new lc lib test"
}
l = Loader(d)

for i in range(6):
    config.L2_LAMBDA = 0.3**i
    print("\n CLS train {:.2f} lambda \n".format(config.L2_LAMBDA))
    with apply_graph(five_layers_lrelu).as_default():
        train.adaptive_train(20000)

