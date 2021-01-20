#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/10 10:51
@File:          generator.py
'''

from keras.layers import *
from keras.initializers import RandomNormal

def generator_model(x, out_imgsize=(28, 28, 1)):
    out_h, out_w, out_c = out_imgsize
    h, w, = out_h // 4, out_w // 4

    x = Dense(h * w * 224, kernel_initializer=RandomNormal())(x)
    x = Reshape((h, w, 224))(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = UpSampling2D()(x)

    x = Conv2D(112, 5, padding='same', use_bias=False, kernel_initializer=RandomNormal())(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = UpSampling2D()(x)

    x = Conv2D(out_c, 5, padding='same', kernel_initializer=RandomNormal())(x)
    x = Activation('tanh')(x)

    return x
