#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/16 6:52
@File:          data_generator.py
'''

import numpy as np

def data_generator(dataset, batch_size=64, shuffle=True, drop_last=False):
    image = dataset[0]
    image_size = image.shape

    true_examples = len(dataset)
    rd_index = np.arange(true_examples)
    false_examples = true_examples // batch_size * batch_size
    remain_examples = true_examples - false_examples

    i = 0
    while True:
        real_batch_size = batch_size
        if remain_examples != 0 and drop_last is False and i == false_examples:
            real_batch_size = remain_examples

        batch_images = np.empty((real_batch_size, *image_size), dtype='float32')

        for b in range(real_batch_size):
            if shuffle and i == 0:
                np.random.shuffle(rd_index)
                dataset.images = dataset.images[rd_index]

            batch_images[b] = dataset[i]

            if remain_examples != 0 and drop_last is True:
                i = (i + 1) % false_examples
            else:
                i = (i + 1) % true_examples

        yield batch_images