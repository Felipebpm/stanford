import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
        self.theta = self.theta.reshape(self.theta.shape[0], 1)
        cost = np.sum((np.dot(X, self.theta) - y)**2) / 2
        print("Cost: " + str(cost))
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        feature_map = X
        for i in range(k - 1):
            feature_map = np.concatenate((feature_map, np.array([X.T[1]**(i + 2)]).T), axis=1)
        return feature_map

        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        X_0 = self.create_poly(k, X)
        return np.concatenate((X_0, np.array([np.sin(X.T[1])]).T), axis=1)
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.dot(X, self.theta)
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        clf = LinearModel()
        if sine:
            X = clf.create_sin(k, train_x)
        else:
            X = clf.create_poly(k, train_x)
        clf.fit(X, train_y)
        x_test = np.array([plot_x[:, 1]]).T
        X_0 = np.zeros(x_test.shape) + 1
        x_concat = np.concatenate((X_0, x_test), axis=1)
        if sine:
            X_test = clf.create_sin(k, x_concat)
        else:
            X_test = clf.create_poly(k, x_concat)
        plot_y = clf.predict(X_test)
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()

def main(train_path, small_path, eval_path):
    '''
    Run all experiments
    '''
    # *** START CODE HERE ***
    run_exp(train_path, ks=[3])
    run_exp(train_path, ks=[3, 5, 10, 20])
    run_exp(train_path, sine=True, ks=[0, 1, 2, 3, 5, 10, 20])
    run_exp(small_path, ks=[0, 1, 2, 3, 5, 10, 20], filename='plot-overfit.png')
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
