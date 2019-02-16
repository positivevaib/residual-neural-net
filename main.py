# import dependencies
import argparse
import numpy as np
import os

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers

import dnn

# create argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type = int, default = 30, help = 'total number of epochs')

args = parser.parse_args()

# load and prepare data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# normalize data
x_train = (x_train - x_train.mean()) / x_train.std()
x_test = (x_test - x_test.mean()) / x_test.std()

# convert labels to categorical matrices
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# create session
sess = tf.Session()
K.set_session(sess)

# train neural net
model = dnn.Net()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x = x_train, y = y_train, batch_size = 128, epochs = 100, shuffle = True, validation_data = (x_test, y_test))