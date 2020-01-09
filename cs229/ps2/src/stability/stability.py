# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import matplotlib.pyplot as plt


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    count, _ = X.shape

    print(-X.dot(theta)[0])
    probs = 1. / (1 + np.exp(-X.dot(theta)))
    grad = (Y - probs).dot(X)

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    theta = np.zeros(X.shape[1])
    learning_rate = 0.1

    i = 0
    w = []
    idx = []
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta + learning_rate * grad
        if i % 500 == 0:
            w += [np.linalg.norm(theta)]
            idx += [i]
        if i % 10000 == 0:
            #  learning_rate /= i
            print('Finished %d iterations' % i)
            m = X.shape[0]
            probs = 1. / (1 + np.exp(-X.dot(theta)))
        if np.linalg.norm(prev_theta - theta) < 1e-15 or i > 100000:
            print('Converged in %d iterations' % i)
            break
    plt.plot(idx, w)
    plt.ylabel(r'$||\theta||_2$', size=11)
    plt.xlabel('iterations')
    plt.savefig('plot_w.png')
    return theta


def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
    theta = logistic_regression(Xa, Ya)
    util.plot(Xa, Ya, theta, "plot_a.png")

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)
    theta = logistic_regression(Xb, Yb)
    print(theta)
    util.plot(Xb, Yb, theta, "plot_b.png")


if __name__ == '__main__':
    main()
