# import dependencies
import argparse
import numpy as np
import os

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.utils as utils

# create argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type = int, default = 200, help = 'total number of epochs')
parser.add_argument('-n', '--nb_res_blocks', type = int, default = 3, help = 'number of residual blocks for each feature map size')

args = parser.parse_args()

# load and prepare data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# normalize data
x_train = (x_train - x_train.mean()) / x_train.std()
x_test = (x_test - x_test.mean()) / x_test.std()

# convert labels to categorical matrices
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# define model
def id_block(x, tot_filters, kernel_size):
    '''residual block with identity shortcut connection'''
    x_orig = x
    x = layers.Conv2D(tot_filters, (kernel_size, kernel_size), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.0001))(x)
    x = layers.Conv2D(tot_filters, (kernel_size, kernel_size), padding = 'same', kernel_regularizer = regularizers.l2(0.0001))(x)
    x = layers.add([x, x_orig])
    x = layers.ReLU()(x)

    return x

def proj_block(x, tot_filters, kernel_size):
    '''residual block with projection shortcut connection'''
    x_orig = x
    x = layers.Conv2D(tot_filters, (kernel_size, kernel_size), strides = (2, 2), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.0001))(x)
    x = layers.Conv2D(tot_filters, (kernel_size, kernel_size), padding = 'same', kernel_regularizer = regularizers.l2(0.0001))(x)
    x_orig = layers.Conv2D(tot_filters, (1, 1), strides = (2, 2), padding = 'same', kernel_regularizer = regularizers.l2(0.0001))(x_orig)
    x = layers.add([x, x_orig])
    x = layers.ReLU()(x)

    return x

def res_stack(x, n, tot_filters, kernel_size, proj = False):
    '''stack of residual blocks'''
    if not proj:
        for _ in range(2*n):
            x = id_block(x, tot_filters, kernel_size)
    else:
        x = proj_block(x, tot_filters, kernel_size)
        for _ in range(2*n - 1):
            x = id_block(x, tot_filters, kernel_size)
    
    return x

def resnet(input_shape, n):
    '''residual neural network'''
    x_init = layers.Input(input_shape)
    x = layers.Conv2D(16, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.0001))(x_init)
    x = res_stack(x, n, 16, 3)
    x = res_stack(x, n, 32, 3, proj = True)
    x = res_stack(x, n, 64, 3, proj = True)
    x = layers.AveragePooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation = 'softmax')(x)

    model = models.Model(inputs = x_init, outputs = x)

    return model

# create session
sess = tf.Session()
K.set_session(sess)

# instantiate model
model = resnet((32, 32, 3), args.nb_res_blocks)
model.summary()

# compile model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# train model
model.fit(x = x_train, y = y_train, batch_size = 128, epochs = args.epochs, validation_data = (x_test, y_test))

# save model
model.save('resnet.h5')