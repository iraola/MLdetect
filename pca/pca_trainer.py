import os

import numpy as np
import pandas as pd
import logging
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .incrementalpca_detect import IncrementalPCA_detect
from teutils import fetch_datasets, HornParallelAnalysis, plot_roc_curve

logger = logging.getLogger(__name__)


class PCA_trainer:
    """
    Pipeline class to preprocess, train and output fault detection results
    using IncrementalPCA (for large datasets).

    Typical usage:
        preprocess > fit > postprocess
    """

    def __init__(self, w_size, train_filepaths, drop_tags, model_name,
                 sample_time=36, alpha=0.01, n_components=None,
                 statistics=('t2', 'spe'), debug=False):
        logger.info('*' * 30)
        logger.info(f'Initializing PCA_trainer object '
                    f'- {model_name}')

        self.results_df = None
        self.model = None
        self.results = None
        self.scaler = None
        self.n_components = n_components
        self.statistics = statistics
        self.w_size = w_size
        self.train_filepaths = train_filepaths
        self.drop_tags = drop_tags
        self.debug = debug
        self.alpha = alpha
        self.model_name = model_name
        self.sample_time = sample_time

        # Generate list of metric names
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
        metric_names += [f'Accuracy IDV{idv}' for idv in range(21)
                         if idv != 15]
        metric_names += [f'Delay IDV{idv}' for idv in range(1, 21)
                         if idv != 15]
        self.metric_names = metric_names

    def preprocess(self):
        """
        Perform data fetching, relabelling, standard scaling and calculation of
        the number of principal components to be retained based on parallel
        analysis.

        Returns
        -------
        scaler : sklearn StandardScaler object
        n_components : int

        """
        logger.info('Starting preprocessing procedure...')

        # Init
        w_size = self.w_size
        drop_tags = self.drop_tags
        train_filepaths = self.train_filepaths

        # Load whole train data in memory (only NOC)
        X_train = fetch_datasets(train_filepaths, drop_list=drop_tags,
                                 identifier='', k=1, M=w_size, shuffle=False,
                                 verbose=0, n=None)
        y_train = X_train.pop('fault')
        y_train[y_train != 0] = 1

        # Center and normalize data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Calculate principal components hyperparameter
        if self.n_components is None:
            n_components = 0
            for i in range(100):
                # Arbitrary number of tries for parallel analysis determination
                n_components, flag = HornParallelAnalysis(X_train_scaled, K=30)
                if not flag:
                    break
            self.n_components = n_components

        self.scaler = scaler
        return self.scaler, self.n_components

    def fit(self, adjust_th=True):
        """
        Fit data using iterations per each training file.

        The method adjust the thresholds by default taking advantage of the
        data accumulated during the iterations and calling `adjust_threshold_2`

        Parameters
        ----------
        adjust_th : bool, optional
            Whether to adjust the threshold to a specific accuracy.
            The default is True.

        Returns
        -------
        model : models.pca.incrementalpca_detect.IncrementalPCA_detect
            The model fitted.

        """
        logger.info('Starting fit procedure...')

        # Init
        train_filepaths = self.train_filepaths
        debug = self.debug
        alpha = self.alpha
        n_components = self.n_components

        # Create object and fit
        model = IncrementalPCA_detect(
            n_components, alpha, statistics=self.statistics)

        # Test-time option
        if debug:
            filepaths = train_filepaths[:10]
        else:
            filepaths = train_filepaths

        # Loop for partial fits
        y_list = []
        X_scaled_list = []
        for filepath in filepaths:
            # Each batch is a single file
            X_scaled, y = self._load_and_preprocess_file(filepath, label=None,
                                                         shuffle=True)
            # Accumulate data
            if adjust_th:
                X_scaled_list.append(X_scaled)
                y_list.append(y)
            # Perform partial fit
            model.partial_fit(X_scaled)

        print(
            f'Number of components with {sum(model.explained_variance_ratio_)}'
            f' variance ratio retained: {model.n_components_}'
        )

        # Concatenate data and adjust threshold
        if adjust_th:
            X_scaled = np.concatenate(X_scaled_list, axis=0)
            y = pd.concat(y_list, axis=0)
            for stat in self.statistics:
                model.adjust_threshold_2(X_scaled, y, stat, acc_target=0.995)

        self.model = model

        return model

    def postprocess(self, my_dir, stat, verbose=0, plot=False, dst_dir=''):
        """
        Get metrics of performance.

        Parameters
        ----------
        my_dir : str
            Directory with files containing the data over which the model is to
            be tested.
        stat : str
            Statistic: 't2' or 'spe'.
        verbose : int, optional
            To allow or not print statements. The default is 0.
        plot : bool, optional
            To allow or not plots (mainly for delays). The default is False.
        dst_dir : str
            Directory in which to store plots. The default is an empty string

        Returns
        -------
        results_df : pandas DataFrame
            Result of the performance analysis: dataframe containing several
            performance metrics.

        """
        logger.info('Starting postprocessing procedure...')
        # Init
        metric_names = self.metric_names
        model = self.model
        dataset_name = my_dir.split(os.sep)[-1]

        # Get decision function scores
        y_truth, y_scores = self._pca_scores(my_dir, stat, label=None,
                                             verbose=0)
        # Get prediction by applying a threshold (binary)
        y_pred = model.predict(X=None, stat=stat, score=y_scores)

        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(y_truth, y_scores)
        filename = f'roc_w{self.w_size}_{stat}_{dataset_name}'
        plot_roc_curve(fpr, tpr, fig_path=os.path.join(dst_dir, filename))

        # Get Metrics
        acc = accuracy_score(y_truth, y_pred)
        prec = precision_score(y_truth, y_pred)
        recall = recall_score(y_truth, y_pred)
        f1 = f1_score(y_truth, y_pred)
        if len(set(y_truth)) > 1:
            roc_auc = roc_auc_score(y_truth, y_scores)
        else:
            roc_auc = None
        # Verbose
        if verbose:
            print('Overall accuracy:', acc)
            print('Precision:', prec)
            print('Recall:', recall)
            print('F1 score:', f1)
            print('ROC area under the curve:', roc_auc)

        # Per class metrics
        idv_list = list(range(21))
        idv_list.remove(15)
        acc_class = np.zeros((20,))
        delays_class = []  # np.zeros((20,))
        for i, idv in enumerate(idv_list):
            # Accuracies
            y_truth_class, y_scores_class = self._pca_scores(
                my_dir, stat, label=idv, verbose=0)
            y_pred_class = model.predict(X=None, stat=stat,
                                         score=y_scores_class)
            acc_class[i] = accuracy_score(y_truth_class, y_pred_class)
            # Detection delays
            if idv != 0:
                mean_delay, max_delay = self._get_delays(my_dir, stat,
                                                         label=idv, plot=plot)
                delays_class.append(mean_delay)
            else:
                mean_delay = np.nan
                max_delay = np.nan
                delays_class.append(np.nan)
            # Verbose
            if verbose:
                print(f'Accuracy for class {idv}: {acc_class[i]}')
                if idv != 0:
                    print(f'Delays for class {idv}:',
                          f'mean = {mean_delay} min',
                          f'max = {max_delay} min')

        # Accumulate results (WARNING: keep same order as in `metric_names`)
        results = [acc, prec, recall, f1, roc_auc] + acc_class.tolist() \
            + delays_class[1:]
        self.results = results

        # Store results
        dataset_name = (my_dir.split(os.sep)[-1]).capitalize()
        results_df = pd.DataFrame(results, index=metric_names,
                                  columns=[dataset_name])
        self.results_df = results_df
        return results_df

    def _pca_scores(self, my_dir, stat, label=None, verbose=0):
        """
        Loop over data in `my_dir` and evaluate model.

        Use "label" to obtain the scores for a specific label only.

        Parameters
        ----------
        my_dir : str
            Directory with data for evaluation.
        stat : str
            Statistic.
        label : int, optional
            Fault class number to narrow down the evaluation.
            The default is None.
        verbose : int, optional
            The default is 0.

        Returns
        -------
        y_truth : array-like
            Actual labels.
        y_scores : array-like
            Scores obtained using the model's decision function. `y_scores` is
            not an array of labels but numbers to be thresholded to obtain
            predicted classes.

        """
        # Init
        logger.info('Running PCA_trainer._pca_scores method.')
        model = self.model
        debug = self.debug
        n_total = 0
        y_truth = []
        y_scores = []

        for file in os.listdir(my_dir):
            # Load and preprocess data
            filepath = os.path.join(my_dir, file)
            X_scaled, y = self._load_and_preprocess_file(filepath, label)
            if X_scaled is None and y is None:
                # The file did not contain the correct label
                continue
            y_truth.append(y)
            n_total += len(X_scaled)

            # Compute score
            y_scores.append(model.decision_function(X_scaled, stat))

            # Test time
            if debug and (n_total > 5000):
                # Leave loop early for testing purposes
                break

        if y_scores:
            y_scores = np.concatenate(y_scores, axis=0)
        if y_truth:
            y_truth = pd.concat(y_truth, axis=0)
        return y_truth, y_scores

    def _get_delays(self, my_dir, stat, label=None, plot=False):
        """
        Loop over data in `my_dir` assuming each file is a scenario to
        calculate the delay in fault detection. Meant to be used specifying
        `label`, therefore returning the mean and max value of the delay for
        each fault class.

        Parameters
        ----------
        my_dir : str
        stat : str
        label : int, optional
            The default is None.
        plot : bool, optional
            The default is False.

        Returns
        -------
        delays.mean()
        delays.max()

        """
        # Init
        logger.info('Running PCA_trainer._get_delays method.')
        model = self.model
        sample_time = self.sample_time
        debug = self.debug
        column_example = 6  # 'XMEAS(7)' for pandas dataframe
        delays = []

        # Method
        i = 0
        for file in os.listdir(my_dir):
            # Load and preprocess data
            filepath = os.path.join(my_dir, file)
            X_scaled, y_truth = self._load_and_preprocess_file(filepath, label)
            if X_scaled is None and y_truth is None:
                # The file did not contain the correct label
                continue
            else:
                i += 1

            # Predict
            y_scores = model.decision_function(X_scaled, stat)
            y_pred = model.predict(X=None, stat=stat, score=y_scores)

            # Visualize
            if plot:
                self.plot_scenario(X_scaled[:, column_example], y_truth,
                                   y_pred)
            delays.append(self.measure_delay(y_truth, y_pred,
                                             timestep=sample_time))

            # Test time
            if debug and (i >= 1):
                # Leave loop early for testing purposes
                break

        delays = np.array(delays)
        if len(delays) != 0:
            mean = delays.mean()
            max = delays.max()
        else:
            mean = np.nan
            max = np.nan
        return mean, max

    def _load_and_preprocess_file(self, filepath, label=None, shuffle=False):
        """ Lower level method to extract data for each file at a time. """
        # Init
        logger.info('Running PCA_trainer._load_and_preprocess_file method.')
        drop_tags = self.drop_tags
        scaler = self.scaler
        w_size = self.w_size

        # Load and preprocess data
        X = fetch_datasets([filepath], drop_list=drop_tags, identifier='',
                           k=1, M=w_size, shuffle=False, verbose=0, n=None)
        y = X.pop('fault')

        # Focus on specific label if specified
        if label is not None:
            if label in y.tolist():
                X = X[y == label]
                y = y[y == label]
            else:
                return None, None
        y[y != 0] = 1

        # Scale data
        X_scaled = scaler.transform(X)

        return X_scaled, y

    def measure_delay(self, y_truth, y_pred, steps2detect=6, timestep=36):
        """
        Return the delay between the occurrence of a fault and its first
        detection. It calls itself recursively upon an early false positive.

        The detection is considered completed when `step2detect` steps occur
        sequentially, the default being 6 as suggested by L. H. Chiang, E. L.
        Russell, and R. D. Braatz, “Results and Discussion,” in Fault
        Detection and Diagnosis in Industrial Systems, London: Springer
        London, 2001, pp. 121–172.

        To do this, the predicted data series is convoluted with a 1D kernel of
        ones and size `steps2detect`. Therefore, the detection index
        corresponds to the first occurrence (np.where()) of the number
        `steps2detect` in the convolution

        Valid only for scenarios with a single fault occurrence.

        Parameters
        ----------
        y_truth : array-like
            truth label values.
        y_pred : array-like
            predicted label values.
        steps2detect : int, optional
            Number of consecutive timesteps until the detection is considered
            accomplished. The default is 6
        timestep : int, optional
            sampling time int the data used. The default is 36.

        """
        logger.info('Running PCA_trainer.measure_delay method.')
        # Get first occurrence of a fault both in truth and the predicted state
        start_index_truth = np.where(y_truth)[0][0]
        # Check if the model predicts any fault at all, otherwise return NaN
        if np.any(y_pred):
            kernel = np.ones((steps2detect,))
            convolved_series = np.convolve(y_pred, kernel)
            try:
                start_index_pred = \
                    np.where(convolved_series == steps2detect)[0][0] \
                    - (steps2detect - 1)
            except:
                # if np.where does not return a tuple it will mean there is no
                # occurrence of the full sequence so no "formal" detection
                # appear in the prediction
                return np.nan
                # start_index_pred = np.where(y_pred)[0][0]
        else:
            return np.nan

        index_diff = start_index_pred - start_index_truth

        if index_diff >= 0:
            # Return delay in minutes
            return index_diff * timestep / 60
        else:
            # Ignore early false positive and recursively call this function
            return self.measure_delay(y_truth[start_index_pred + 1:],
                                      y_pred[start_index_pred + 1:], timestep)

    def plot_scenario(self, X, y_truth, y_pred, timestep=36):
        """
        Plot a scenario truth and predicted values to visualize how good
        the predictor is.
        """
        # Init (time in hours)
        logger.info('Running PCA_trainer.plot_scenario method.')
        fig, ax = plt.subplots(figsize=(12, 8))
        time = np.arange(timestep, timestep * (len(X) + 1), timestep) / 3600
        # Plot
        ax.plot(time, X, linewidth=2, label='Process variable')
        ax1 = ax.twinx()
        ax1.plot(time, y_truth, 'g', linewidth=2, label='Plant state')
        ax1.plot(time, y_pred, 'r.', label='Predicted state')
        # Format
        ax.legend(loc='best')
        ax1.legend(loc='best')
