#!/usr/bin/env python

import argparse
import h5py
import numpy as np

DATA_DATASET = 'matrixset'
DIAGONALS_DATASET = 'diagonalset'
WINDOW = 10

def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        metavar='DATA',
        type=str,
        dest='data',
        help='path to the HDF5 file that stores the matrices'
    )

    return parser.parse_args()


def extract_diagonals(path):
    handle = h5py.File(path, 'r+')
    # load the data and move the sample axis to the front
    data = np.moveaxis(np.array(handle[DATA_DATASET]), -1, 0)

    width = data.shape[1]
    samples = data.shape[0]
    diagonalset = np.zeros((samples, 2 * WINDOW + 1, width), dtype=np.float32)

    for j in range(samples):
        image = data[j, :, :]
        # always reallocate the diagonals image here to fill the left triangle with ones
        out = np.ones((2 * WINDOW + 1, width), dtype=np.float32)
        for i in range(-WINDOW, WINDOW + 1):
            diagonal = np.diagonal(image, i)
            out[i + WINDOW, abs(i):] = diagonal
        out = ((out - out.min()) / out.max() * 2) - 1
        diagonalset[j] = out

    # remove the previous diagonals set if present
    if DIAGONALS_DATASET in list(handle.keys()):
        del handle[DIAGONALS_DATASET]
    handle[DIAGONALS_DATASET] = diagonalset

    # release the handle
    handle.close()


if __name__ == '__main__':
    arguments = parse_cli()
    extract_diagonals(arguments.data)

