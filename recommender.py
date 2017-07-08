import logging
import sys

import numpy as np
from scipy.optimize import minimize


class Recommender:

    DEFAULT_NUM_FEATURES = 10
    DEFAULT_REGULARIZATION = 1
    DEFAULT_MAX_ITER = 100

    def __init__(self, num_features=DEFAULT_NUM_FEATURES,
                 reg=DEFAULT_REGULARIZATION, Y=None, R=None):
        self.num_features = num_features
        self.reg = reg
        self.Y = Y
        self.R = R
        self.Theta = None
        self.X = None

    def learn(self, Y=None, R=None, num_features=None, reg=None, verbose=False,
              maxiter=DEFAULT_MAX_ITER):
        import utils

        # Set the variables first
        if Y is not None:
            self.Y = Y
        if R is not None:
            self.R = R
        if reg is not None:
            self.reg = reg
        if num_features is not None:
            self.num_features = num_features

        # Prepare all the required local variables
        Y, R, num_features, reg = self.Y, self.R, self.num_features, self.reg
        num_movies, num_users = Y.shape

        # Initialize random parameters
        logging.info("Initializing random paramters...")
        initial_X = np.random.randn(num_movies, num_features)
        initial_Theta = np.random.randn(num_users, num_features)
        initial_params = np.append(initial_X.flatten(), initial_Theta.flatten())
        extra_args = (Y, R, num_features, reg)

        def callback(x):
            if verbose:
                sys.stdout.write('.')
                sys.stdout.flush()

        # Cost minimization
        logging.info("Running the optimizer...")
        result = minimize(
            fun=utils.cf_cost,
            x0=initial_params,
            args=extra_args,
            jac=True,
            method='CG',
            callback=callback,
            options={
                'maxiter': maxiter,
                'disp': verbose,
            })

        # Extract learned parameters
        self.X = np.reshape(result.x[:num_movies*num_features], (num_movies, num_features))
        self.Theta = np.reshape(result.x[num_movies*num_features:], (num_users, num_features))

        return self

    def save(self, filename):
        import utils
        logging.info("Saving recommender model to '{}'".format(filename))
        utils.save_to_file(self, filename)

    def load(filename):
        import utils
        logging.info("Loading recommender model from '{}'".format(filename))
        return utils.load_from_file(filename)
