#!/usr/bin/env python

import argparse
import pickle
import h5py
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, add, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, ZeroPadding2D
from keras.models import load_model, Model
from keras.regularizers import l2
from keras.utils import plot_model
import numpy as np


def positive_int(value):
    try:
        parsed = int(value)
        if not parsed > 0:
            raise ValueError()
        return parsed
    except ValueError:
        raise argparse.ArgumentTypeError('value must be an positive integer')


def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-e', '--epochs',
        nargs='?',
        type=positive_int,
        action='store',
        default=10,
        help='number of training epochs'
    )
    parser.add_argument(
        metavar='TRAIN',
        type=str,
        dest='train',
        help='path to the HDF5 file with the training data'
    )
    parser.add_argument(
        metavar='MODEL',
        type=str,
        dest='model',
        help='path where to store the model'
    )

    return parser.parse_args()


def load_data(path):
    with h5py.File(path, 'r') as handle:
        data = np.array(handle['diagonalset'])
        labels = np.array(handle['vectorset'])

        return data, labels


def preprocess(data, labels):
    # simply add an additional dimension for the channels for data
    # swap axis of the label set
    return np.expand_dims(data, axis=3), np.moveaxis(labels, 0, -1)


def build_model(input_shape):
    input_img = Input(shape=input_shape)

    # first bottleneck unit
    bn_1 = BatchNormalization()(input_img)
    activation_1 = Activation('selu')(bn_1)
    conv_1 = Conv2D(32, kernel_size=(5, 5,), padding='same', kernel_regularizer=l2(0.02))(activation_1)

    bn_2 = BatchNormalization()(conv_1)
    activation_2 = Activation('selu')(bn_2)
    conv_2 = Conv2D(128, kernel_size=(3, 3,), padding='same', kernel_regularizer=l2(0.02))(activation_2)

    merged = add([input_img, conv_2])

    # corner detection
    bn_3 = BatchNormalization()(merged)
    padding = ZeroPadding2D(padding=(0, 3))(bn_3)
    conv_3 = Conv2D( 32, kernel_size=(21, 7,), padding='valid', activation='tanh')(padding)
    conv_4 = Conv2D(128, kernel_size=( 1, 3,), padding='same',  activation='tanh')(conv_3)
 
    # fully-connected predictor
    flat = Flatten()(conv_4)
    classify = Dense(512, activation='sigmoid')(flat)
    dropout = Dropout(0.1)(classify)

    result = Dense(input_shape[1], activation='sigmoid')(dropout)

    model = Model(inputs=input_img, outputs=result)
    model.compile(optimizer=optimizers.Nadam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train_network(model, data, labels, model_file, epochs):
    plot_model(model, to_file='{}.png'.format(model_file), show_shapes=True)

    checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=True, save_best_only=True, save_weights_only=False, mode='auto')
    training = model.fit(data, labels, epochs=epochs, batch_size=8, validation_split=1.0/5.0, class_weight={0: 0.1, 1: 0.9}, callbacks=[checkpoint])

    with open('{}.history'.format(model_file), 'wb') as handle:
        pickle.dump(training.history, handle)


if __name__ == '__main__':
    arguments = parse_cli()
    data, labels = preprocess(*load_data(arguments.train))
    model = build_model(input_shape=data.shape[1:])
    train_network(model, data, labels, arguments.model, arguments.epochs)

