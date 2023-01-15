# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:26:15 2022

@author: Eduardo.Iraola
"""
import logging
import time

import numpy as np
import pandas as pd
from math import copysign

from matplotlib import pyplot as plt
from scipy.stats import f, chi2
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PCA_detect(PCA):
    """Inherits from sklearn PCA class"""

    def __init__(
            self,
            n_components=None,
            alpha=0.05,
            *,
            statistics=('t2', 'spe'),
            copy=True,
            whiten=False,
            svd_solver="auto",
            tol=0.0,
            iterated_power="auto",
            random_state=None,
    ):
        """Call original __init__ and instantiate alpha"""
        logger.info('Running PCA_detect.__init__ method.')
        # Parameter initializations
        self.spe_th = None
        self.t2_th = None
        self.statistics = statistics
        # Call parent __init__
        logger.info("Initializing sklearn's PCA object")
        super().__init__(
            n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            random_state=random_state,
        )
        self.alpha = alpha

    def __str__(self):
        return 'PCA_detect object'

    def __repr__(self):
        return self.__str__()

    def fit(self, X, y=None):
        """
        Fit the model with X and add scaling and statistic preprocessing to the
        standard PCA.fit

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        logger.info('Running PCA_detect.fit method.')
        self._fit(X)

        return self

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X."""
        logger.info('Running PCA_detect.fit_transform method.')

        U, S, Vt = self._fit(X)
        U = U[:, : self.n_components_]

        # X_new = X * V = U * S * Vt * V = U * S
        # U *= S[: self.n_components_] # Substituted code
        U = U.dot(S[: self.n_components_])

        # Adjust to compensate for the pretransformation previous to 'fit'
        U *= np.sqrt(self.n - 1)

        return U

    def _fit(self, X):
        """Common code to fit and fit_transform."""
        logger.info('Running PCA_detect._fit method.')

        # Init and preprocessing
        n = len(X)

        # Run sklearn's PCA fit (IMPORTANT: 1 / np.sqrt(n-1))
        U, S, Vt = super()._fit(1 / np.sqrt(n - 1) * X)
        # Get the matrix of retained singular values
        S = np.diag(self.singular_values_)

        # Compute full singular values (PCA only gets "preserved" ones)
        # s_full is a vector with values in descending order
        _, s_full, _ = np.linalg.svd(1 / np.sqrt(n - 1) * X,
                                     full_matrices=True)

        # Precalculate `np.linalg.inv(S.dot(S))` to get the t2 stat later
        t1 = time.perf_counter()
        S_square_inv = np.linalg.inv(S.dot(S))
        t2 = time.perf_counter()
        logger.info(f'>> Elapsed time to compute S^-2: {(t2 - t1) / 60} min')

        self.n = n  # Number of training NOC instances
        # self.m = X.shape[1]  # Number of original features
        self.S = S
        self.s_full = s_full
        self.S_square_inv = S_square_inv

        # Calculate thresholds for detection
        t2_th, spe_th = self.calc_threshold()
        self.t2_th = t2_th
        self.spe_th = spe_th

        return U, S, Vt

    def calc_threshold(self, verbose=0):
        """Calculate statistic thresholds for detection"""
        logger.info('Running PCA_detect.calc_threshold method.')

        # Init
        n = self.n
        q = self.n_components_
        s_full = self.s_full
        eigenvalues = s_full ** 2

        if 't2' in self.statistics:
            # HOTELLING'S T2 STATISTIC
            # Calculate F tail value for given confidence interval (1-alpha)
            # (PPF = Percent Point Function)
            F = f.ppf(1 - self.alpha, dfn=q, dfd=n - q)
            t2_th = (q * (n - 1) * (n + 1)) / (n * (n - q)) * F
        else:
            t2_th = None
            F = None

        if 'spe' in self.statistics:
            # Q STATISTIC OR SPE (SQUARED PREDICTION ERROR)
            # Compute threshold according to
            #   Nomikos and MacGregor, Multivariate SPC charts for monitoring
            #   batch processes, 1995 (found in Heo and Lee, 2019)
            theta_1 = np.sum(eigenvalues[q:])
            theta_2 = np.sum(eigenvalues[q:] ** 2)
            g = theta_2 / theta_1
            h = theta_1 ** 2 / theta_2
            spe_th = g * chi2.ppf(1 - self.alpha, df=h, loc=0, scale=1)
        else:
            spe_th = None

        if verbose:
            print('F dist. value for n =', n - q, 'and a =', q, ': F =', F)
            print('The T^2 statistic threshold is:', t2_th)
            print('The SPE statistic threshold is:', spe_th)

        return t2_th, spe_th

    def set_threshold(self, value, stat):
        """ Sets a specific statistic threshold """
        logger.info('Running PCA_detect.set_threshold method.')
        if stat == 't2':
            self.t2_th = value
        elif stat == 'spe':
            self.spe_th = value

    def get_threshold(self, stat):
        """ Sets a specific statistic threshold """
        logger.info('Running PCA_detect.get_threshold method.')
        if stat == 't2':
            return self.t2_th
        elif stat == 'spe':
            return self.spe_th

    def predict(self, X, stat, score=None):
        """
        Predict binary classification result of X based on statistic specified
        in 'stat'

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) (raw data)
            Data to make predictions on, where `n_samples` is the number of
            samples and `n_features` is the number of features.

        stat : str.
            Name of the statistic to be used for prediction.
            Options are 't2' and 'spe'

        score : array-like. Optional
            Used if provided to avoid calling decision_function. Useful
            if X is large to fit in memory for SVD. Default: None

        Returns
        -------
        y_pred : object
            Returns the instance itself.
        """
        logger.info('Running PCA_detect.predict method.')

        # Get matrix of transformed data, X_transformed
        if score is None:
            score = self.decision_function(X, stat)

        if stat == 't2':
            threshold = self.t2_th
        elif stat == 'spe':
            threshold = self.spe_th
        else:
            raise Exception(f'statistic {stat} not implemented')

        # Evaluate results on threshold and convert to integer
        y_pred = score > threshold
        return y_pred.astype(int)

    def decision_function(self, X, stat):
        """ Predict confidence scores for samples. """
        logger.info('Running PCA_detect.decision_function method.')
        # Get matrix of transformed data, X_transformed
        if stat == 't2':
            X_transformed = self.transform(X)
            T2_score = self.t2_statistic(X_transformed)
            return T2_score
        elif stat == 'spe':
            SPE_score = self.spe_statistic(X)
            return SPE_score

    def t2_statistic(self, X_transformed):
        """ Get T2 statistics from transformed instances. """
        logger.info('Running PCA_detect.t2_statistic method.')

        # Without loop (too much memory)
        # T2 = X_transformed.dot(np.linalg.inv(S.dot(S))).dot(X_transformed.T)
        # T2 = np.diag(T2)

        T2 = np.empty((len(X_transformed),))
        T2[:] = np.NaN
        for i, x in enumerate(X_transformed):
            T2[i] = x.dot(self.S_square_inv).dot(x.T)

        return T2

    def spe_statistic(self, X):
        """ Get SPE statistics from transformed instances. """
        logger.info('Running PCA_detect.spe_statistic method.')

        # Initializations
        m = self.n_features_
        P = self.components_

        # Without loop (too much memory)
        # r = (np.identity(m) - (P.T).dot(P)).dot(X.T)
        # SPE = np.diag(r.T.dot(r))

        # Calculate residuals (vectors length 'm', see Chiang's)
        SPE = np.empty((len(X),))
        SPE[:] = np.NaN
        for i, x in enumerate(X):
            r = (np.identity(m) - (P.T).dot(P)).dot(x.T)
            SPE[i] = r.T.dot(r)

        # #****************************
        # P_R = self.components_[self.n_components_:]
        # print('vectors corresponding to residuals:\n', P_R)
        # r = X_transformed.dot(P_R.T)
        # Q = np.diag(r.dot(r.T))
        # print('Q values:\n', Q)
        # SPE = Q
        # #****************************

        return SPE

    def adjust_threshold(self, X, K=10):
        """
        Adjust thresholds to the K'th highest value (Chiang et al.) based
        on a set of data.
        """
        logger.info('Running PCA_detect.adjust_threshold method.')

        # Check if input data is a dataframe
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy()

        # Get statistics from data
        X_transformed = self.transform(X)
        T2 = self.t2_statistic(X_transformed)
        SPE = self.spe_statistic(X)

        # Sort values descendingly and set thresholds
        t2_th_new = np.flip(np.sort(T2))[K - 1]
        spe_th_new = np.flip(np.sort(SPE))[K - 1]
        print(f'New T2 threshold is {t2_th_new}. Old was {self.t2_th}'
              f'\nNew SPE threshold is {spe_th_new}. Old was {self.spe_th}')
        self.t2_th = t2_th_new
        self.spe_th = spe_th_new

    def score(self, X, y, stat):
        """ Evaluate prediction on X data. """
        logger.info('Running PCA_detect.score method.')
        # Check if input data is a dataframe
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy()
        # Perform prediction
        y_pred = self.predict(X, stat)
        # Evaluate with respect to the ground truth values
        score = accuracy_score(y, y_pred)

        return score

    def adjust_threshold_2(self, X_train_scaled, y_train, stat,
                           acc_target=0.995):
        """
        Apply bisection method to adjust stat's threshold to obtain a specific
        accuracy over the train set NOC instances.
        """
        logger.info('Running PCA_detect.adjust_threshold_2 method.')

        # Init
        # model = self.model
        max_iter = 1000
        tol = 1e-4

        y_scores = self.decision_function(X_train_scaled, stat=stat)
        y_pred = self.predict(X=None, stat=stat, score=y_scores)
        acc = accuracy_score(y_train, y_pred)
        initial_th = self.get_threshold(stat)
        test_offset = 0.01 * abs(initial_th)

        if acc > acc_target:
            # We want to go to the left
            test_offset = - test_offset
            r_th = initial_th
        elif acc < acc_target:
            # We want to go to the right
            test_offset = test_offset
            l_th = initial_th

        # Get the other point for the bisection method
        i = 0
        while True:
            # Calculate and set new threshold
            new_th = test_offset + initial_th
            self.set_threshold(new_th, stat)

            # Test it on the scores obtained from the model
            y_pred = self.predict(X=None, stat=stat, score=y_scores)
            acc_new = accuracy_score(y_train, y_pred)
            print(
                f'Testing threshold {new_th} with train accuracy = {acc_new}'
            )
            if copysign(1, acc_new - acc_target) == copysign(1,
                                                             acc - acc_target):
                # Condition still not reached.
                #  Exponentially increase the offset: 2, 4, 8, 16, ...
                test_offset += test_offset
            else:
                break

            # Exhausting iterations
            i += 1
            if i >= max_iter:
                print('Convergence criteria not reached, exiting loop')
                break

        if acc > acc_target:
            l_th = new_th
        elif acc < acc_target:
            r_th = new_th
        print(f'Original threshold: {initial_th}, new threshold:',
              f'{self.get_threshold(stat)}')

        # Bisection procedure
        i = 0
        print('Starting bisection procedure for statistic', stat)
        while abs((acc_new - acc_target) / acc_target) > tol:
            # Calculate and set new threshold
            new_th = (l_th + r_th) / 2
            self.set_threshold(new_th, stat)

            # Test it on the scores obtained from the model
            y_pred = self.predict(X=None, stat=stat, score=y_scores)
            acc_new = accuracy_score(y_train, y_pred)
            if acc_new > acc_target:
                r_th = new_th
            else:
                l_th = new_th
            print(f'Testing threshold {new_th} with accuracy = {acc_new}')

            # Exhausting iterations
            i += 1
            if i >= max_iter:
                print('Convergence criteria not reached')
                break


def plot_iris(X, target_names, n, pca):
    # Plot dataset
    fig, ax = plt.subplots(figsize=[10, 6])
    ax.plot(X[:n, 0], X[:n, 1], 'b.', label=target_names[0])
    ax.plot(X[n:2 * n, 0], X[n:2 * n, 1], 'rx', label=target_names[1])
    ax.plot(X[2 * n:, 0], X[2 * n:, 1], 'gd', label=target_names[2])
    ax.legend(loc='best')
    ax.set_xlim([None, None])
    ax.set_ylim([None, None])

    # Plot detection boundaries
    n_mesh = 100
    x1 = np.linspace(-5, 25, n_mesh)
    x2 = np.linspace(-5, 25, n_mesh)
    x1v, x2v = np.meshgrid(x1, x2)
    t2 = np.zeros((n_mesh, n_mesh))
    for i in range(n_mesh):
        for j in range(n_mesh):
            t = np.array([x1v[i, j], x2v[i, j]])
            t = t[:, np.newaxis].T
            # print(t)
            t2[i, j] = pca.t2_statistic(t)

    # Do the contour plots to check the boundaries of the SPC method
    CS = ax.contour(x1v, x2v, t2, levels=[pca.t2_th])
    ax.clabel(CS, CS.levels, inline=True, fontsize=10)
    plt.show()


def main_iris_detect():
    """
    Test the PCA_detect class using the iris dataset. This shows how this class
    should be used, and it is mainly the base for the PCA_trainer class, adding
    some other useful tools along the way.

    """
    # Init parameters
    alpha = 0.05
    n_components = 2

    # Get data
    iris = load_iris()
    X = iris.data
    y = iris.target
    y[y != 0] = 1
    X_noc = X[y == 0]

    # preprocess
    scaler = StandardScaler()
    X_noc = scaler.fit_transform(X_noc)
    X = scaler.transform(X)

    # Create object and fit
    pca = PCA_detect(n_components, alpha)
    pca.fit(X_noc)

    # Evaluate model
    print('T2 Score:', pca.score(X, y, stat='t2'))
    print('SPE Score:', pca.score(X, y, stat='spe'))

    # Plot
    pca.components_[1, :] = -pca.components_[1, :]  # just for prettier plot
    X_reduced = pca.transform(X)
    plot_iris(X_reduced, target_names=iris.target_names, n=len(X_noc), pca=pca)


def main_get_statistics():
    """
    Show how to use PCA_detect just to obtain T2 statistics from a dataset.

    Even though it uses PCA, it fits the data using all the features, so it
    does not exactly do dimensionality reduction. But for the
    `decision_function` method to take place, we first need to compute the
    matrix S, and the fit method does that.

    Used as a base for Gaussian_NN.

    """
    # Init
    alpha = 0.05
    iris = load_iris()
    X = iris.data
    y = iris.target
    y[y != 0] = 1
    X_noc = X[y == 0]
    n_components = X.shape[1]

    # Scaling
    scaler = StandardScaler()
    scaler.fit_transform(X_noc)
    X = scaler.transform(X)

    # PCA class object usage
    pca = PCA_detect(n_components=n_components, alpha=alpha)
    pca.fit(X_noc)
    t2 = pca.decision_function(X, stat='t2')

    # Show results
    plt.hist(t2[:len(X_noc)], bins=20)
    plt.title('NOC statistic values')
    plt.figure()
    plt.hist(t2[len(X_noc):], bins=40)
    plt.title('Anomaly statistic values')
    plt.show()


if __name__ == '__main__':
    # main_iris()
    main_get_statistics()
