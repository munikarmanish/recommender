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

    for i in range(num_movies):
        idx = np.where(R[i, :] == 1)[0]  # users who have rated movie i
        temp_theta = Theta[idx, :]  # parameter vector for those users
        temp_Y = Y[idx, :]  # ratings given to movie i
        X_grad[i, :] = np.sum(np.dot(np.dot(temp_theta, X[i, :]) - temp_Y.T,
                                     temp_theta) + reg * X[i, :], axis=0)

    for j in range(num_users):
        idx = np.where(R[:, j] == 1)[0]
        temp_X = X[idx, :]
        temp_Y = Y[idx, j]
        Theta_grad[j, :] = np.sum(np.dot(np.dot(Theta[j], temp_X.T) -
                                         temp_Y, temp_X) + reg * Theta[j], axis=0)

    grad = np.append(X_grad.flatten(), Theta_grad.flatten())
    return (J, grad)
