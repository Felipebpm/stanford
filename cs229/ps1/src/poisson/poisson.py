import numpy as np
import math
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    theta = np.random.randn(x_train.shape[1], 1) * 0.001
    clf = PoissonRegression(theta_0=theta)
    clf.fit(x_train, y_train)
    # Run on the validation set, and use np.savetxt to save outputs to save_path
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    predictions = clf.predict(x_eval)
    util.plot(predictions, y_eval, clf.theta, "poisson-plot.jpg", correction=1.0)
    np.savetxt(save_path, predictions)
    # *** END CODE HERE ***

def log_vector_factorial(x):
    a = np.array(map(np.math.factorial, x)).reshape(x.shape)
    return np.array(map(math.log, a)).reshape(x.shape)


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        m = x.shape[0]
        y = y.reshape((m, 1))
        theta_prev = self.theta
        for i in range(self.max_iter):
            eta = np.dot(x, self.theta)
            h_theta = np.exp(eta)
            logfact = log_vector_factorial(y)
            cost = np.sum(np.exp(np.dot(-1, eta)) + np.multiply(y, eta) - logfact)
            d_theta = np.dot(x.T, (y - h_theta))
            assert(d_theta.shape == self.theta.shape)
            #  print(d_theta.shape)
            self.theta = self.theta + self.step_size * d_theta
            if self.verbose:
                print("Cost after iteration " + str(i) + ": " + str(cost))
            if np.linalg.norm(self.step_size * d_theta) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.exp(np.dot(x, self.theta))
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')

