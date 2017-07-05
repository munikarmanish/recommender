#!/bin/env python3

import logging
import sys
import unittest

import numpy as np
from scipy.io import loadmat

import utils


class RecommenderTest(unittest.TestCase):

    def test_initialization(self):
        return


class CostFunctionTest(unittest.TestCase):

    def test_numerical_gradient(self):
        x = np.random.random((3, 3))
        grad = 2 * x
        numgrad = utils.numerical_grad(lambda x: np.sum(x**2), x)
        np.testing.assert_almost_equal(grad, numgrad)

    def test_cf_cost(self):
        # logging.info("Loading dataset...")
        R = utils.load_from_file('data/R.bin')
        Y = utils.load_from_file('data/Y.bin')

        # logging.info("Loading pre-trained parameters...")
        _ = loadmat('data/movie_params.mat')
        X = _.get('X')
        Theta = _.get('Theta')

        # reduce dataset
        num_users = 4
        num_movies = 5
        num_features = 3
        X = X[:num_movies, :num_features]
        Theta = Theta[:num_users, :num_features]
        Y = Y[:num_movies, :num_users]
        R = R[:num_movies, :num_users]

        params = np.append(X.flatten(), Theta.flatten())
        cost = utils.cf_cost(params=params, Y=Y, R=R, num_features=num_features, reg=0)[0]
        # logging.info("Expected cost = 22.22")
        # logging.info("Computed cost = {:.2f}".format(cost))
        self.assertAlmostEqual(22.22, cost, places=2)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr,
                        format='[%(levelname)s]: %(message)s', level=logging.INFO)
    unittest.main()
