import pickle

import numpy as np

from recommender import Recommender


def load_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_to_file(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def numerical_grad(f, x, h=1e-8):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        index = it.multi_index
        original = x[index]
        x[index] = original + h
        f_high = f(x)
        x[index] = original - h
        f_low = f(x)
        x[index] = original
        grad[index] = (f_high - f_low) / (2 * h)
        it.iternext()
    return grad


def cf_cost(params, Y, R, num_features=Recommender.DEFAULT_NUM_FEATURES,
            reg=Recommender.DEFAULT_REGULARIZATION):
    # Unpack the parameters
    num_movies, num_users = Y.shape
    num_params = num_movies * num_features
    X = params[:num_params].reshape((num_movies, num_features))
    Theta = params[num_params:].reshape((num_users, num_features))

    J = 0
    X_grad = np.zeros_like(X)
    Theta_grad = np.zeros_like(Theta)

    J = (.5 * np.sum(((np.dot(Theta, X.T).T - Y) * R)**2) +
         ((reg / 2) * np.sum(Theta**2)) +
         ((reg / 2) * np.sum(X**2)))

    for l in range(num_movies):
        for b in range(num_features):
            X_grad[l, b] = 0
            for allJ in range(num_users):
                if R[l, allJ] == 1:
                    X_grad[l, b] += (np.dot(Theta[allJ, :], X[l, :].T) -
                                     Y[l, allJ]) * Theta[allJ, b]
            X_grad[l, b] += reg * X[l, b]

    for nU in range(num_users):
        for c in range(num_features):
            Theta_grad[nU, c] = 0
            for allI in range(num_movies):
                if R[allI, nU] == 1:
                    Theta_grad[nU, c] += (np.dot(Theta[nU, :], X[allI, :].T) -
                                          Y[allI, nU]) * X[allI, c]
            Theta_grad[nU, c] = Theta_grad[nU, c] + (reg * Theta[nU, c])

    grad = np.append(X_grad.flatten(), Theta_grad.flatten())
    return (J, grad)
