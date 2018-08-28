#!/usr/bin/env python

import argparse
import h5py
import keras.models
import numpy as np


def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-m', '--model',
        metavar='MODEL',
        type=str,
        default='model.h5',
        dest='model_file',
        help='path to the model checkpoint file'
    )
    parser.add_argument(
        metavar='FILE',
        type=str,
        dest='predict_file',
        help='path to the HDF5 file with the prediction data'
    )

    return parser.parse_args()


def load_data(path):
    with h5py.File(path, 'r') as handle:
        data = np.array(handle['diagonalset'])
        labels = np.array(handle['vectorset'])

        return data, labels


def load_model(model_file):
    return keras.models.load_model(model_file)

def preprocess(data, labels):
    # simply add an additional dimension for the channels for data
    # swap axis of the label set
    return np.expand_dims(data, axis=3), np.moveaxis(labels, 0, -1)


def predict(data, model):
    return model.predict(data, batch_size=1, verbose=True)


def store(prediction, path):
    prediction_dataset = 'predictionset'
    with h5py.File(path, 'r+') as handle:
        if prediction_dataset in handle:
            del handle[prediction_dataset]
        handle[prediction_dataset] = prediction


if __name__ == '__main__':
    arguments = parse_cli()

    data, labels = preprocess(*load_data(arguments.predict_file))
    model = load_model(arguments.model_file)
    prediction = predict(data, model)
    store(prediction, arguments.predict_file)
