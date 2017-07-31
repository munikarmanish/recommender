#!/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open('log.csv'), delimiter=',')

def plot_reg_rmse():

    reg = data[0:11, 1]
    rmse = data[0:11, 2]
    mae = data[0:11, 3]

    fig = plt.figure()
    plt.plot(reg, rmse)
    plt.plot(reg, mae)
    plt.xscale('log')
    plt.grid(linestyle=':')
    plt.axis([1e-6, 1e4, 0, 4])
    plt.xlabel('Regularization parameter')
    plt.ylabel('Error')
    plt.legend(['RMSE', 'MAE'])
    #plt.show()
    fig.savefig('reg-v-rmse.pdf', bbox_inches='tight', pad_inches=0.1)

def plot_reg_cost():

    reg = data[0:11, 1]
    cost = data[0:11, 4]

    fig = plt.figure()
    plt.plot(reg, cost, linewidth=2)
    plt.xscale('log')
    plt.grid(linestyle=':')
    plt.axis([1e-6, 1e4, 0, 700000])
    plt.xlabel('Regularization parameter')
    plt.ylabel('Cost')
    #plt.show()
    fig.savefig('reg-v-cost.pdf', bbox_inches='tight', pad_inches=0.1)

def plot_feature_rmse():

    x = data[11:, 0]
    rmse = data[11:, 2]
    mae = data[11:, 3]

    fig = plt.figure()
    plt.plot(x, rmse)
    plt.plot(x, mae)
    plt.grid(linestyle=':')
    plt.axis([5, 50, 1.8, 2.5])
    plt.xlabel('Feature vector dimension')
    plt.ylabel('Error')
    plt.legend(['RMSE', 'MAE'])
    #plt.show()
    fig.savefig('feature-v-rmse.pdf', bbox_inches='tight', pad_inches=0.1)

def plot_feature_cost():

    x = data[11:, 0]
    cost = data[11:, 4]

    fig = plt.figure()
    plt.plot(x, cost, linewidth=2)
    plt.grid(linestyle=':')
    plt.axis([5, 50, 65000, 75000])
    plt.xlabel('Feature vector dimension')
    plt.ylabel('Cost')
    #plt.show()
    fig.savefig('feature-v-cost.pdf', bbox_inches='tight', pad_inches=0.1)

if __name__ == '__main__':
    plot_reg_rmse()
    plot_reg_cost()
    plot_feature_rmse()
    plot_feature_cost()
