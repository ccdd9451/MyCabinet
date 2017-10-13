#!env python
# coding: utf-8
# Dep: learn for crowd2
# at - 4aaaa62

import tensorflow as tf
from lc import train, config, analysis, Loader
from tensorflow.contrib.layers import fully_connected, summarize_collection
from tensorflow.contrib.keras.python.keras.layers import LeakyReLU

config.NUM_UNIT = 10
config.DATASIZE = 256
config.STOP_THRESHOLD = 10**-8
config.VERBOSE_EACH = 100


def max_out(inputs, num_units=config.NUM_UNIT, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(
                             num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs




def five_layers(x, ref_y, test):
    test = None if not test else True
    activation = max_out
    hid1 = fully_connected(
        x, 1000, activation_fn=activation, reuse=test, scope="layer1")
    hid2 = fully_connected(
        hid1, 1000, activation_fn=activation, reuse=test, scope="layer2")
    hid3 = fully_connected(
        hid2, 1000, activation_fn=activation, reuse=test, scope="layer3")
    hid4 = fully_connected(
        hid3, 1000, activation_fn=activation, reuse=test, scope="layer4")
    hid5 = fully_connected(
        hid4, 1000, activation_fn=activation, reuse=test, scope="layer5")
    y = fully_connected(
        hid5, 1, activation_fn=tf.identity, reuse=test, scope="fc")
    if not test:
        analysis.add_RMSE_loss(y, ref_y, "train")
        analysis.add_L2_loss()
    else:
        analysis.add_RMSE_loss(y, ref_y, "test")


def apply_graph(graph, BGD=True):
    g1 = tf.Graph()
    with g1.as_default():
        with tf.name_scope("train_net"):
            if BGD:
                x1, y1 = l.shuffle_batch(batch_size=config.DATASIZE)
            else:
                x1, y1 = l.train()

            graph(x1, y1, False)

        with tf.name_scope("test_net"):
            x2, y2 = l.validation()
            graph(x2, y2, True)

        summarize_collection("rates")
        # summarize_collection("visuals")
    return g1


def screen_logic():
    config.RESTORE_FROM = None
    config.L2_LAMBDA = 0.00
    config.LEARNING_RATE = 0.08
    config.DECAY_RATE = 0.95
    config.DECAY_STEP = 50
    with apply_graph(five_layers, BGD=False).as_default():
        return train.adaptive_train(5000)

def train_logic():
    config.L2_LAMBDA = 0.00
    config.LEARNING_RATE = 0.001
    config.DECAY_RATE = 0.96
    config.DECAY_STEP = 200
    with apply_graph(five_layers, BGD=True).as_default():
        config.RESTORE_FROM, *_ = train.adaptive_train(5000)

    config.L2_LAMBDA = 0.01
    config.LEARNING_RATE = 0.0005
    config.DECAY_RATE = 0.96
    config.DECAY_STEP = 200
    with apply_graph(five_layers, BGD=True).as_default():
        config.RESTORE_FROM, *_ = train.adaptive_train(5000)

    config.L2_LAMBDA = 0.02
    config.LEARNING_RATE = 0.0005
    config.DECAY_RATE = 0.96
    config.DECAY_STEP = 200
    with apply_graph(five_layers, BGD=True).as_default():
        train.adaptive_train(5000)

config.DATAFILE = "learn_len.dat"
d = {"name": "LenScreen", "discription": "Brute force to attemp converge."}
l = Loader(d)
details = []
for i in range(100):
    name = "lenScreen"
    details.append(screen_logic())

d = sorted(
    details,
    key = lambda x: x[1][1]
)

config.DATAFILE = "learn_len.dat"
d = {"name": "LenScrAdv", "discription": "Attemp to converge."}
l = Loader(d)
for i in d[:10]:
    config.RESTORE_FROM = i[0]
    train_logic()

