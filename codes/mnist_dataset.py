'''
@Author:        ZM
@Date and Time: 2020/7/26 14:37
@File:          mnist_dataset.py
'''

import numpy as np
import gzip

def get_mnist(root_dir='D:/datasets/MNIST'):
    train_images_path = f'{root_dir}/train-images-idx3-ubyte.gz'
    train_labels_path = f'{root_dir}/train-labels-idx1-ubyte.gz'

    val_images_path = f'{root_dir}/t10k-images-idx3-ubyte.gz'
    val_labels_path = f'{root_dir}/t10k-labels-idx1-ubyte.gz'

    with gzip.open(train_images_path, 'rb') as f:
        X_train = np.frombuffer(f.read(), dtype='uint8', offset=16).reshape(-1, 28, 28)

    with gzip.open(train_labels_path, 'rb') as f:
        Y_train = np.frombuffer(f.read(), dtype='uint8', offset=8)

    with gzip.open(val_images_path, 'rb') as f:
        X_val = np.frombuffer(f.read(), dtype='uint8', offset=16).reshape(-1, 28, 28)

    with gzip.open(val_labels_path, 'rb') as f:
        Y_val = np.frombuffer(f.read(), dtype='uint8', offset=8)

    return (X_train, Y_train), (X_val, Y_val)