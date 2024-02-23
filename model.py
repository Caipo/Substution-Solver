
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, \
Concatenate, Input, Conv1DTranspose, Dropout, Dense, Reshape, Flatten, Softmax

from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from tensorflow.keras.models import Model


def convolution_block(kernal, input_):
    path_a = Dense(1000, activation=None)(input_)
    path_a = Reshape((1000, 1))(path_a)
    path_a = Conv1D(64, kernal, data_format = 'channels_last')(path_a)
    path_a = MaxPooling1D(pool_size=(4), strides=1, padding='valid')(path_a)
    path_a = Conv1D(64, kernal, data_format = 'channels_last')(path_a)
    path_a = MaxPooling1D(pool_size=(4), strides=1, padding='valid')(path_a)
    return Flatten()(path_a)


def cracker(size):
    input_ = Input((size), dtype = 'float32')

    main_path = Dense(1000, activation=None)(input_)

    path_a = convolution_block(1, main_path)
    path_b = convolution_block(2, main_path)
    path_c = convolution_block(3, main_path)
    path_d = convolution_block(4, main_path)

    main_path = Concatenate()([path_a, path_b, path_c, path_d])
        
    main_path = Dense(10000, activation='relu')(main_path)
    main_path = Dense(5000, activation='relu')(main_path)
    main_path = Dense(2500, activation='relu')(main_path)
    main_path = Dense(1000, activation='relu')(main_path)
    main_path = Dense(676, activation='relu')(main_path)
    main_path = Reshape((26, 26))(main_path)
    main_path = Softmax(axis=1)(main_path)

    return Model(inputs = input_, outputs=main_path)

