#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/1 20:12
@File:          train.py
'''

import math
import numpy as np
import cv2
from keras.layers import Input
from keras import Model
from keras.optimizers import RMSprop

from Dataset import Dataset
from mnist_dataset import get_mnist
from data_generator import data_generator
from discriminator_model import discriminator_model
from generator_model import generator_model

if __name__ == '__main__':
    num_classes = 1
    batch_size = 64
    d_init_lr = 1e-5
    gan_init_lr = 1e-5
    epochs = 300
    initial_epoch = 0
    img_size = (28, 28, 1)
    dst_img_size = (140, 140)
    latent_dim = 100

    (X_train, Y_train), _ = get_mnist()
    X_train = X_train[Y_train == 8]
    X_train = np.expand_dims(X_train / 127.5 - 1, 3)

    dataset = Dataset(X_train)
    generator = data_generator(dataset, batch_size=batch_size, shuffle=True)

    d_input = Input(shape=img_size, dtype='float32')
    d_out = discriminator_model(d_input)
    d_model = Model(d_input, d_out)
    opt = RMSprop(learning_rate=d_init_lr, decay=1e-8)
    d_model.compile(opt, loss='binary_crossentropy')

    g_input = Input(shape=(latent_dim, ), dtype='float32')
    g_out = generator_model(g_input, out_imgsize=img_size)
    g_model = Model(g_input, g_out)

    d_model.trainable = False
    gan_input = Input(shape=(latent_dim, ), dtype='float32')
    gan_out = d_model(g_model(gan_input))
    gan_model = Model(gan_input, gan_out)

    opt = RMSprop(learning_rate=gan_init_lr, decay=1e-8)
    gan_model.compile(opt, loss='binary_crossentropy')

    num_batches = math.ceil(len(X_train) / batch_size)
    last_epoch = initial_epoch + epochs
    best_d_loss = best_gan_loss = math.inf

    for i_epoch in range(initial_epoch, last_epoch):
        print(f'Epoch {i_epoch + 1}/{last_epoch}')

        total_d_loss = total_a_loss = 0.

        for i_batch in range(num_batches):
            real_batch_images = next(generator)
            batch_size = len(real_batch_images)
            random_latent_vectors = np.random.randn(batch_size, latent_dim)
            generated_images = g_model.predict_on_batch(random_latent_vectors)
            combined_images = np.concatenate([real_batch_images, generated_images], axis=0)
            labels = np.concatenate([np.ones((batch_size, 1), dtype='float32'),
                                     np.zeros((batch_size, 1), dtype='float32')], axis=0)
            index = np.arange(batch_size * 2)
            np.random.shuffle(index)
            combined_images = combined_images[index]
            labels = labels[index]

            d_model.trainable = True
            d_loss = d_model.train_on_batch(combined_images, y=labels)

            random_latent_vectors = np.random.randn(batch_size * 2, latent_dim)
            misleading_targets = np.ones((batch_size * 2, 1), dtype='float32')

            d_model.trainable = False
            a_loss = gan_model.train_on_batch(random_latent_vectors, y=misleading_targets)

            total_d_loss += d_loss
            total_a_loss += a_loss

            print(f'\rd_loss = {d_loss:.12f}, a_loss = {a_loss:.12f}', end='', flush=True)

        print(f'd_loss = {total_d_loss / num_batches:.12f}, a_loss = {total_a_loss / num_batches:.12f}')

        if total_d_loss < best_d_loss:
            d_model.save_weights('best_d_wts.weights')

        if total_a_loss < best_gan_loss:
            g_model.save_weights('best_g_wts.weights')

        img = cv2.resize(np.round((real_batch_images[0] + 1) * 127.5).astype('uint8'), dst_img_size)
        cv2.imwrite('real_image.png', img)

        img = cv2.resize(np.round((generated_images[0] + 1) * 127.5).astype('uint8'), dst_img_size)
        cv2.imwrite('generated_image.png', img)
