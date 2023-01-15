import logging
import time

import numpy as np
from sklearn.decomposition import IncrementalPCA

from .pca_detect import PCA_detect

logger = logging.getLogger(__name__)


class IncrementalPCA_detect(PCA_detect, IncrementalPCA):
    """Inherits from sklearn Incremental PCA class"""

    def __str__(self):
        return 'IncrementalPCA_detect object'

    def partial_fit(self, X, y=None, check_input=True):
        """
        Calls super().partial_fit() inherited from IncrementalPCA class, but
        modifies the input matrix to fit the adequate sqrt(n-1) normalization
        to apply process monitoring and later stores the full S matrix, not
        only the values retained by PC.

        THE AUTOMATIC CALCULATION OF THE SPE THRESHOLD MAY NOT BE ACCURATE DUE
        TO THE MANUAL RE-CALCULATION OF S_FULL.

        It is recommended to use large batches or readjust the threshold after
        fit

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        check_input : bool, default=True
            Run check_array on X.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        logger.info('Running IncrementalPCA.partial_fit method.')
        n = len(X)
        super().partial_fit(1 / np.sqrt(n - 1) * X, y=None, check_input=True)

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
        self.n_features_ = X.shape[1]  # Number of original features
        self.S = S
        self.s_full = s_full
        self.S_square_inv = S_square_inv

        # TODO: we should calculate covariance matrices in a cumulative way
        #  since currently we are just keeping the last partial_fit one

        # Calculate thresholds for detection
        t2_th, spe_th = self.calc_threshold()
        self.t2_th = t2_th
        self.spe_th = spe_th

        return self
