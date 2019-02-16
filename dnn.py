# import dependencies
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers

# define model
class Net(tf.keras.Model):
    '''resnet'''
    def __init__(self):
        '''constructor'''
        super(Net, self).__init__()
        self.conv1 = layers.Conv2D(64, (8, 8), strides = (2, 2), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.05))
        self.max_pool = layers.MaxPool2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer =  regularizers.l2(0.05))
        self.conv3 = layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'valid', activation = 'relu', kernel_regularizer =  regularizers.l2(0.05))
        self.zero_pad1 = layers.ZeroPadding2D((4, 4))
        self.conv4 = layers.Conv2D(128, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer =  regularizers.l2(0.05))
        self.conv5 = layers.Conv2D(256, (3, 3), strides = (2, 2), padding = 'valid', activation = 'relu', kernel_regularizer =  regularizers.l2(0.05))
        self.zero_pad2 = layers.ZeroPadding2D((2, 2))
        self.conv6 = layers.Conv2D(256, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer =  regularizers.l2(0.05))
        self.avg_pool = layers.AveragePooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(10, activation = 'softmax')

    def call(self, inputs):
        '''forward pass'''
        x1 = self.max_pool()(self.conv1(inputs))
        x2 = self.conv2(self.conv2(x1)) + x1
        x3 = self.conv2(self.conv2(x2)) + x2
        x4 = self.conv2(self.conv2(x3)) + x3
        x5 = self.conv3(x4)
        x6 = self.conv4(x5) + self.zero_pad1(x4)
        x7 = self.conv4(self.conv4(x6)) + x6
        x8 = self.conv4(self.conv4(x7)) + x7
        x9 = self.conv4(self.conv4(x8)) + x8
        x10 = self.conv5(x9)
        x11 = self.conv6(x10) + self.zero_pad2(x9)
        x12 = self.conv6(self.conv6(x11)) + x11
        x13 = self.conv6(self.conv6(x12)) + x12
        x14 = self.conv6(self.conv6(x13)) + x13
        x15 = self.conv6(self.conv6(x14)) + x14
        x16 = self.avg_pool()(self.conv6(self.conv6(x15)) + x15)
        y = self.dense(self.flatten(x16))

        return y