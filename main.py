#!/bin/env python3

from matplotlib import pyplot as plt
from scipy.io import loadmat


def main():
    print('[INFO] Loading matrix...')
    raw_mat = loadmat('movies.mat')
    # R = raw_mat.get('R')
    Y = raw_mat.get('Y')

    # print('R:', R.shape)
    # print('Y:', Y.shape)

    print('[INFO] Visualizing rating matrix...')
    plt.matshow(Y)
    plt.xlabel("Users")
    plt.ylabel("Movies")
    plt.show()


if __name__ == '__main__':
    main()
