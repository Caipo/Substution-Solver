
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, \
Concatenate, Input, Conv1DTranspose, Dropout, Dense, Reshape, Flatten, Softmax
from tensorflow.keras.models import Model


def convolution_block(kernal, input_):
    main_path = Dense(1000, activation=None)(input_)
    main_path = Reshape((1000, 1))(main_path)
    main_path = Conv1D(64, kernal, data_format = 'channels_last')(main_path)
    main_path = MaxPooling1D(pool_size=(4), strides=1, padding='valid')(main_path)
    main_path = Conv1D(64, kernal, data_format = 'channels_last')(main_path)
    main_path = MaxPooling1D(pool_size=(4), strides=1, padding='valid')(main_path)
    return Flatten()(main_path)


def cracker(size):
    input_ = Input((size), dtype = 'float32')

    #main_path = Dense(1000, activation=None)(input_)

    path_a = convolution_block(1, input_)
    path_b = convolution_block(2, input_)
    path_c = convolution_block(3, input_)

    main_path = Concatenate()([input_, path_a, path_b, path_c])
    
    starting_size = 5000 

    i = 1

    while starting_size // i > 676 or i >= 3: 

        main_path = Dense(starting_size // i, activation='relu')(main_path)
        i += 1

    main_path = Dense(676, activation='relu')(main_path)
    main_path = Reshape((26, 26))(main_path)
    main_path = Softmax(axis=1)(main_path)

    return Model(inputs = input_, outputs=main_path)

