import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # *** START CODE HERE ***
    # Train a logistic regression classifier
    theta = np.zeros((x_train.shape[1], 1))
    clf = LogisticRegression(theta_0 = theta)
    clf.fit(x_train, y_train)
    # Plot decision boundary on top of validation set set
    x_test, y_test = util.load_dataset(valid_path, add_intercept=True)
    predictions = clf.predict(x_test)
    util.plot(x_test, y_test, clf.theta, "plot.jpg", correction=1.0)
    # Use np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path, predictions)
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        m = x.shape[0]
        y = y.reshape((m, 1))
        theta_prev = self.theta
        for i in range(self.max_iter):
            h_theta = sigmoid(np.dot(x, self.theta))
            cost = - np.sum(np.dot(y.T, np.log(h_theta - (h_theta - 0.5) * self.eps)) + (np.dot((1 - y).T, np.log(1 - h_theta + (h_theta - 0.5) * self.eps)))) / m
            d_theta = np.dot(x.T, (h_theta - y)) / m
            self.theta = self.theta - self.step_size * d_theta
            if self.verbose:
                print("Cost after iteration " + str(i) + ": " + str(cost))
            if np.linalg.norm(self.step_size * d_theta) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        prediction = sigmoid(np.dot(x, self.theta))
        return prediction
        # *** END CODE HERE ***

def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
