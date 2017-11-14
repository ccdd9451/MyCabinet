#!env python
# coding: utf-8
# Dep: learn for crowd2
# at - 69899c3

import tensorflow as tf
import time, datetime
from lc import train, config, analysis, Loader
from tensorflow.contrib.layers import fully_connected, summarize_collection
from xilio import dump

config.NUM_UNIT = 10
config.DATASIZE = 256
config.STOP_THRESHOLD = 10**-8
config.VERBOSE_EACH = 500


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
    for i in range(6):
        x = fully_connected(
            x, 1000, activation_fn=activation, reuse=test,
            scope="layer"+str(i))
    y = fully_connected(
        x, 1, activation_fn=tf.identity, reuse=test, scope="fc")
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
                x1, y1 = nestedData.shuffle_batch(batch_size=config.DATASIZE)
            else:
                x1, y1 = nestedData.train()

            graph(x1, y1, False)

        with tf.name_scope("test_net"):
            x2, y2 = nestedData.validation()
            graph(x2, y2, True)

        summarize_collection("rates")
        # summarize_collection("visuals")
    return g1


def screen_logic():
    config.RESTORE_FROM = None
    config.L2_LAMBDA = 0.00
    config.LEARNING_RATE = 0.01
    config.DECAY_RATE = 0.90
    config.DECAY_STEP = 50
    with apply_graph(five_layers, BGD=False).as_default():
        return train.simple_train(2000)

def train_logic(targetLambda):
    config.L2_LAMBDA = 0.00
    config.LEARNING_RATE = 0.001
    config.DECAY_RATE = 0.96
    config.DECAY_STEP = 200
    with apply_graph(five_layers, BGD=True).as_default():
        config.RESTORE_FROM, *_ = train.adaptive_train(8000)

    config.L2_LAMBDA = targetLambda / 2
    config.LEARNING_RATE = 0.0005
    config.DECAY_RATE = 0.96
    config.DECAY_STEP = 200
    with apply_graph(five_layers, BGD=True).as_default():
        config.RESTORE_FROM, *_ = train.adaptive_train(8000)

    config.L2_LAMBDA = targetLambda
    config.LEARNING_RATE = 0.0005
    config.DECAY_RATE = 0.96
    config.DECAY_STEP = 200
    with apply_graph(five_layers, BGD=True).as_default():
        return train.adaptive_train(8000)


def timeReport():
    global result

    elapsedTime = time.time() - startTime
    print("""
Training round {round} finished, total time using is {timeUsage}.
Estimated finishing time is {estiTime}.
""".format(
        round=len(result),
        timeUsage=str(datetime.timedelta(seconds=elapsedTime)),
        estiTime=time.ctime(
            startTime + elapsedTime / len(result) * 30)))


result = [1]
startTime = time.time()

timeReport()
result = []

config.DATAFILE = "learn_len.dat"
d = {"name": "lenFin", "discription": "massive screen on len"}
nestedData = Loader(d)
for j in [0.05, 0.1]:
    for i in range(5):
        config.RESTORE_FROM, *_ = screen_logic()
        result.append(train_logic(j))
        timeReport()

config.DATAFILE = "learn_time.dat"
d = {"name": "timeLen", "discription": "massive screen on time"}
nestedData = Loader(d)
for j in [0.05, 0.1]:
    for i in range(5):
        config.RESTORE_FROM, *_ = screen_logic()
        result.append(train_logic(j))
        timeReport()

config.DATAFILE = "learn_time.dat"
d = {"name": "clsFin", "discription": "massive screen on cls"}
nestedData = Loader(d)
for j in [0.05, 0.1]:
    for i in range(5):
        config.RESTORE_FROM, *_ = screen_logic()
        result.append(train_logic(j))
        timeReport()


dump("summary.dat", result)
