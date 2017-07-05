#!/bin/env python3

from utils import load_from_file

NUM_FEATURES = 10


def main():
    print('[INFO] Loading matrix...')
    Y = load_from_file('data/Y.bin')
    print('Y:', Y)


if __name__ == '__main__':
    main()
