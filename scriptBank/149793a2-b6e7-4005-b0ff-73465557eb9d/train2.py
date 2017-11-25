#!env python
# coding: utf-8
# Dep: learn for crowd2
# at - 3ae527e

import tensorflow as tf
import time
import datetime
import numpy as np
from pathlib import Path
from lc import train, config, analysis, Loader, graph
from tensorflow.contrib.layers import fully_connected, summarize_collection
from xilio import dump

config.NUM_UNIT = 10
config.DATASIZE = 256
config.STOP_THRESHOLD = 10**-8
config.VERBOSE_EACH = 500


def five_layers(x, ref_y, test):
    test = None if not test else True
    activation = graph.max_out
    for i in range(6):
        x = fully_connected(
            x,
            1000,
            activation_fn=activation,
            reuse=test,
            scope="layer" + str(i))
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


def init_train():
    config.RESTORE_FROM = None
    config.L2_LAMBDA = 0.00
    config.LEARNING_RATE = 0.005
    config.DECAY_RATE = 1
    config.DECAY_STEP = 50
    with apply_graph(five_layers, BGD=False).as_default():
        return train.simple_train(20000)


def train_once(tlmbd):
    config.L2_LAMBDA = tlmbd
    config.LEARNING_RATE = 0.0002
    config.DECAY_RATE = 0.96
    config.DECAY_STEP = 200
    with apply_graph(five_layers, BGD=True).as_default():
        return train.adaptive_train(10000)


def timeReport(data):
    global result
    result.append((config.RESTORE_FROM, data))
    elapsedTime = time.time() - startTime
    print("""
Training round {round} finished, total time using is {timeUsage}.
Estimated finishing time is {estiTime}.
""".format(
        round=len(result),
        timeUsage=str(datetime.timedelta(seconds=elapsedTime)),
        estiTime=time.ctime(startTime + elapsedTime / len(result) * 28)))


def trainOnD():
    config.RESTORE_FROM, *d = init_train()
    timeReport(d)
    config.RESTORE_FROM, *d = train_once(0.001)
    timeReport(d)
    for j in np.geomspace(0.005, 0.1, 5):
        config.RESTORE_FROM, *d = train_once(j)
        timeReport(d)


result = []
startTime = time.time()
print("Project start at ", time.ctime())

for i in map(str, Path(".").glob("*_avg")):
    config.DATAFILE = i
    d = {"name": "d3" + i[:-4], "discription": "massive screen on " + i}
    nestedData = Loader(d)
    trainOnD()

dump("summary2.dat", result)
