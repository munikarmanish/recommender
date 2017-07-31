#!/bin/env python3

import csv

import utils
from recommender import Recommender

DEFAULT_NUM_FEATURES = 10
DEFAULT_REG = 10
DEFAULT_MAX_ITER = 500


def main():
    R = utils.load_from_file('data/R.bin').astype(float)
    Y = utils.load_from_file('data/Y.bin')

    # reg_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    reg_list = [1e3, 1e4]
    num_features_list = [45, 50]

    model = Recommender(Y=Y, R=R)

    # for reg in reg_list:
    #     print("::: Trying reg = {}".format(reg))
    #     model.learn(verbose=True, reg=reg, num_features=DEFAULT_NUM_FEATURES, maxiter=DEFAULT_MAX_ITER)
    #     rmse = model.rmse()
    #     mae = model.mae()
    #     with open("log.csv", "a", newline='') as csvfile:
    #         csvwriter = csv.writer(csvfile)
    #         csvwriter.writerow([DEFAULT_NUM_FEATURES, reg, rmse, mae])

    for num_features in num_features_list:
        print("::: Trying num_feature = {}".format(num_features))
        model.learn(verbose=True, reg=DEFAULT_REG, num_features=num_features, maxiter=DEFAULT_MAX_ITER)
        rmse = model.rmse()
        mae = model.mae()
        with open("log.csv", "a", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([num_features, DEFAULT_REG, rmse, mae])


if __name__ == '__main__':
    main()
