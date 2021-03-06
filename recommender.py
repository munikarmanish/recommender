import sys

import numpy as np
from scipy.optimize import minimize


class Recommender:

    DEFAULT_NUM_FEATURES = 10
    DEFAULT_REGULARIZATION = 1
    DEFAULT_MAX_ITER = 1000

    def __init__(self, num_features=DEFAULT_NUM_FEATURES, normalized=False,
                 reg=DEFAULT_REGULARIZATION, Y=None, R=None, *args, **kwargs):
        self.num_features = num_features
        self.reg = reg
        self.Y = Y
        self.R = R
        self.Theta = None
        self.X = None
        self.normalized = normalized

    def learn(self, Y=None, R=None, num_features=None, reg=None, verbose=False,
              maxiter=DEFAULT_MAX_ITER, normalize=None, tol=1e-2):
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
        if normalize is not None:
            self.normalized = normalize

        # Prepare all the required local variables
        Y, R, num_features, reg = self.Y, self.R, self.num_features, self.reg
        if self.normalized:
            Y, self.Ymean = utils.normalize_ratings(self.Y, self.R)
            self.normalized = True
        num_movies, num_users = Y.shape

        # Initialize random parameters
        if verbose:
            print("Initializing random paramters...")
        initial_X = 0.01 * np.random.randn(num_movies, num_features)
        initial_Theta = 0.01 * np.random.randn(num_users, num_features)
        initial_params = 0.01 * np.append(initial_X.flatten(), initial_Theta.flatten())
        extra_args = (Y, R, num_features, reg)

        def callback(x):
            if verbose:
                sys.stdout.write('.')
                sys.stdout.flush()

        # Cost minimization
        if verbose:
            print("Running the optimizer...")
        result = minimize(
            fun=utils.cf_cost,
            x0=initial_params,
            args=extra_args,
            jac=True,
            method='CG',
            callback=callback,
            tol=tol,
            options={
                'maxiter': maxiter,
                'disp': verbose,
            })

        # Extract learned parameters
        self.X = np.reshape(result.x[:num_movies*num_features], (num_movies, num_features))
        self.Theta = np.reshape(result.x[num_movies*num_features:], (num_users, num_features))

        return self

    def recommendations(self, user_id, n=10):
        Ypredicted = np.dot(self.X, self.Theta.T)
        if self.normalized:
            Ypredicted += self.Ymean.reshape(-1,1)
        user_predictions = Ypredicted[:, user_id].flatten()
        recommended_ids = np.flip(user_predictions.argsort()[-11:], axis=0)
        # movies = utils.load_movie_list('data/movie_ids.txt')
        return [(i, user_predictions[i]) for i in recommended_ids]

    def rmse(self):
        """Root Mean Square Error"""
        Ypredicted = np.dot(self.X, self.Theta.T)
        return np.sqrt(np.mean((self.Y.astype(float) - Ypredicted)**2))

    def mae(self):
        """Mean Absolute Error"""
        Ypredicted = np.dot(self.X, self.Theta.T)
        return np.mean(np.absolute(self.Y.astype(float) - Ypredicted))

    def save(self, filename):
        import utils
        print("Saving recommender model to '{}'".format(filename))
        utils.save_to_file(self, filename)

    def load(filename):
        import utils
        print("Loading recommender model from '{}'".format(filename))
        return utils.load_from_file(filename)
