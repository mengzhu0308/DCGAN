#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/10 11:19
@File:          discriminator_model.py
'''

from keras.layers import *
from keras.initializers import RandomNormal

def discriminator_model(x):
    x = Conv2D(112, 4, strides=2, padding='same', kernel_initializer=RandomNormal())(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(224, 4, strides=2, padding='same', use_bias=False, kernel_initializer=RandomNormal())(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1, kernel_initializer=RandomNormal())(x)
    x = Activation('sigmoid')(x)

    return x
