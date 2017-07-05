#!/bin/env python3

import unittest
import logging
import numpy as np

import utils


class RecommenderTest(unittest.TestCase):

    def test_initialization(self):
        return


class CostFunctionTest(unittest.TestCase):

    def test_numerical_gradient(self):
        x = np.random.random((3,3))
        grad = 2 * x
        numgrad = utils.numerical_grad(lambda x: np.sum(x**2), x)
        np.testing.assert_almost_equal(grad, numgrad)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)10s: %(message)s')
    unittest.main()
