import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    clf = GDA()
    clf.fit(x_train, y_train)
    x_test, y_test = util.load_dataset(valid_path, add_intercept=False)
    predictions = clf.predict(x_test)
    # Plot decision boundary on validation set
    util.plot(x_test, y_test, clf.theta, "plot.jpg", correction=1.0)
    # Use np.savetxt to save outputs from validation set to save_path
    np.savetxt(save_path, predictions)
    # *** END CODE HERE ***

def getsigma(x, mu_0, mu_1, m, y, y_0):
    sigma_ret = np.zeros((x.shape[1], x.shape[1]))
    for i in range(m):
        sigma_0 = np.dot((x[i] - mu_0).T, x[i] - mu_0) * y_0[i]
        sigma_1 = np.dot((x[i] - mu_1).T, x[i] - mu_1) * y[i]
        sigma_ret += sigma_0 + sigma_1
    return sigma_ret / m

def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))

class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """ 
        # *** START CODE HERE ***
        y = y.reshape(y.shape[0], 1)
        y_0 = (1 - y).reshape(y.shape)
        m = y.shape[0]
        m_0 = np.asscalar(np.sum(y_0))
        m_1 = np.asscalar(np.sum(y))
        # Find phi, mu_0, mu_1, and sigma
        phi = np.sum(y) / m
        mu_0 = (np.sum(np.multiply(y_0, x), axis = 0, keepdims = True) / m_0) #.reshape(y.shape)
        mu_1 = np.sum(np.multiply(y, x), axis = 0, keepdims=True) / m_1
        sigma = getsigma(x, mu_0, mu_1, m, y, y_0)
        # Write theta in terms of the parameters
        sigma_inv = np.linalg.inv(sigma)
        log_phi = np.log(np.exp(-1 * np.log(phi)) - 1)
        theta_0 = (np.dot(np.dot(mu_0, sigma_inv), mu_0.T) - np.dot(np.dot(mu_1, sigma_inv), mu_1.T)) / 2 - log_phi
        self.theta = np.concatenate((theta_0, np.dot(sigma_inv, (mu_1 - mu_0).T)))
        # Compute cost
        x_0 = np.zeros((x.shape[0], 1)) + 1
        x_train = np.concatenate((x_0.T, x.T))
        h_theta =  sigmoid(np.dot(self.theta.T, x_train)).T
        cost = - np.sum(np.dot(y.T, np.log(h_theta - (h_theta - 0.5) * self.eps)) + (np.dot(y_0.T, np.log(1 - h_theta + (h_theta - 0.5) * self.eps)))) / m
        if self.verbose:
            print("Cost: " + str(cost))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        x_0 = np.zeros((x.shape[0], 1)) + 1
        x_pred = np.concatenate((x_0.T, x.T))
        return sigmoid(np.dot(self.theta.T, x_pred))
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')

