# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:32:25 2022

@author: Eduardo.Iraola
"""
import logging
import numpy as np
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from teutils import plot_roc_curve, where, fetch_datasets, timed
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, \
    precision_score, recall_score, f1_score
from tensorflow import keras

logger = logging.getLogger(__name__)


def calc_scores(y_true, y_pred, y_scores, scores_list, fig_path):
    """
    Generic method for generation of scores based on y_true, y_pred,
    y_scores.
    """

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    plot_roc_curve(fpr, tpr, fig_path=fig_path)

    # Get Metrics
    scores_dict = {}
    for metric in scores_list:
        if metric.__name__ == 'roc_auc_score':
            unique_values = set(y_true)
            if len(unique_values) > 1:
                scores_dict[metric.__name__] = metric(y_true, y_scores)
            else:
                logger.warning('ROC_AUC metric could not be calculated since'
                               'y_true had only instances of one class')
                scores_dict[metric.__name__] = None
        else:
            scores_dict[metric.__name__] = metric(y_true, y_pred)

    return scores_dict


def identify_fault(y):
    """ Return the number of a fault occurring in an array of labels. """
    unique_values = np.unique(y).tolist()  # type: list
    # Exclude 0 instances if existing in the list
    if 0 in unique_values:
        unique_values.remove(0)
    # Handle multi-fault scenarios (still not supported)
    if len(unique_values) > 1:
        raise NameError('Multi-fault scenarios are still not supported')
    elif len(unique_values) == 0:
        return 0
    else:
        return unique_values[0]


class Base:
    """ Base class with methods for any ML model API """

    def __init__(self, dst_dir=''):

        self.threshold = None
        self.platform = None
        self.drop_tags = None
        self.identifier = None
        self.datasets_dict = None
        self.scores = None
        self.param_dict = None
        self.filepath_dict = None
        self.custom_objects = None
        self.history_dict = None
        self.model_name = None
        self.preprocessing = None
        self.model = None
        self.log_start_section()
        logger.info('Initializing class instance...')

        # Initialize lists of faults
        self.debug = False
        idv_list = list(range(1, 21))  # Exclude 0 (NOC)
        idv_list.remove(15)
        self.idv_list = idv_list
        self.idv_list_noc = [0] + idv_list

        # Initialize lists of metrics
        metric_names = [
            'Accuracy',
            'Precision',
            'Recall',
            'F1',
            'ROC AUC'
        ]
        metric_names += [f'Accuracy IDV{idv}' for idv in range(21)
                         if idv != 15]
        metric_names += [f'Delay IDV{idv}' for idv in range(1, 21)
                         if idv != 15]
        self.metric_names = metric_names
        self.scores_list = [
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score
        ]

        # Other initializations
        self.steps2detect = 6
        self.timestep = 36  # seconds
        self.dst_dir = dst_dir
        self.savefig = True

    def __repr__(self):
        """ Loop through variables in self.__dict__. """
        str_list = ['Model:',
                    '=' * 51,
                    f'{"Parameter name": <25} {"Value" : <25}',
                    '=' * 51]
        # Loop over API instance properties
        for varname, value in self.__dict__.items():
            if varname != 'model_name' and (
                isinstance(value, str)
                or isinstance(value, int)
                or isinstance(value, float)
            ):
                # We don't want to print dictionaries of class
                str_list.append(f'{varname: <25} {value : <25}')
        # Loop over `param_dict` parameters if the instance has this attribute
        if hasattr(self, 'param_dict'):
            for varname, value in self.param_dict.items():
                if varname != 'model_name' and (
                    isinstance(value, str)
                    or isinstance(value, int)
                    or isinstance(value, float)
                ):
                    # We do not want to print dictionaries of complex classes
                    str_list.append(f'{varname: <25} {value : <25}')
            str_list.append('=' * 51)
        return '\n'.join(str_list)

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def log_start_section():
        logger.info('\n\n**************************************')

    def save(self, dst_dir=None):
        # Init
        model = self.model
        model_name = self.model_name
        platform = self.platform

        # Handle different destination directory if provided
        if dst_dir is None:
            dst_dir = self.dst_dir
        # Make directory if it does not exist yet
        if dst_dir and not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)

        # Save model
        model_filepath = os.path.join(dst_dir, model_name)
        logger.info(f'Saving model in {model_filepath}...')
        if platform == 'tensorflow':
            model_filepath = model_filepath + '.h5'
            model.save(model_filepath)
        elif platform == 'torch':
            import torch
            model_filepath = model_filepath + '.pt'
            torch.save(model, model_filepath)
        elif platform == 'sklearn':
            # Save sklearn model as pickle
            model_filepath = model_filepath + '.pkl'
            with open(model_filepath, 'wb') as output:
                pickle.dump(model, output)
        else:
            logger.error(f'Could not save: platform {platform} not identified')
            raise Exception

        # Save auxiliary data - IMPORTANT keep same name in property and string
        extra_data = {
            'preprocessing': self.preprocessing,
            'scores': self.scores,
            'history_dict': self.history_dict,
            'param_dict': self.param_dict
        }
        extra_data_filename = os.path.join(dst_dir, f'{model_name}_dict.pkl')
        logger.info(f'Saving extra data in {extra_data_filename}...')
        with open(extra_data_filename, 'wb') as outp:
            pickle.dump(extra_data, outp)

    def load(self, model_name=None, src_dir=None):
        """
        Load model and extra data (preprocessing, scores, history_dict and
        param_dict) from the files represented by `model_name`.

        By default, use `self.model_name`, but the user can provide another
        string as model name.
        """
        # Init
        custom_objects = self.custom_objects
        platform = self.platform

        # Handle different source if provided
        if not model_name:
            model_name = self.model_name
            load_flag = False
        else:
            load_flag = True

        # Handle directory location
        if src_dir is None:
            src_dir = self.dst_dir
        if not os.path.isdir(src_dir):
            logger.error(f'The directory {src_dir} does not exist. '
                         'Aborting loading procedure')
            raise Exception

        logger.info(f'Loading model and extra data for model {model_name}...')

        # Load auxiliary data
        extra_data_filename = os.path.join(src_dir, model_name + '_dict.pkl')
        if os.path.isfile(extra_data_filename):
            extra_data = pickle.load(open(extra_data_filename, 'rb'))
            # Collect the extra data objects into instance attributes
            for param_name, param_object in extra_data.items():
                setattr(self, param_name, param_object)
        else:
            logger.warning(f'Extra data for model {model_name} not found')
            extra_data = None

        # Platform-specific loading
        if self.platform == 'tensorflow':
            # Load `custom_objects` used to load specific keras objects
            custom_objects = None
            if extra_data:
                if 'param_dict' in extra_data:
                    param_dict = extra_data['param_dict']
                    if 'custom_objects' in param_dict:
                        custom_objects = param_dict['custom_objects']

        # Load model
        model_filepath = os.path.join(src_dir, model_name)
        logger.info(f'Loading model from {model_filepath}...')
        if platform == 'tensorflow':
            model_filepath = model_filepath + '.h5'
            model = keras.models.load_model(model_filepath, custom_objects)
            model.compile(loss=model.loss, optimizer=model.optimizer,
                          metrics=self.param_dict['metrics_list'])
            model.summary(print_fn=logger.info)
        elif platform == 'torch':
            model_filepath = model_filepath + '.pt'
            import torch
            model = torch.load(model_filepath)
            model.eval()
        elif platform == 'sklearn':
            model_filepath = model_filepath + '.pkl'
            model = pickle.load(open(model_filepath, 'rb'))
        else:
            logger.error(f'Could not load: platform {platform} not identified')
            raise Exception
        self.model = model
        
        # Initialize model name if it was provided by the user
        if load_flag:
            self.set_model_name()

        return model, extra_data

    def preprocess(self):
        """ Load and preprocess the data. """
        self.load_data()
        self.standardize_data()

    @timed
    def load_data(self):
        """
        Load datasets from the specified filepaths in filepath_dict directly as
        pandas dataframes. Not suitable for very large datasets since it will
        load them all in memory.
        """
        datasets_dict = {}
        for dataset_name, filepaths in self.filepath_dict.items():
            data = fetch_datasets(
                filepaths, drop_list=self.drop_tags,
                identifier=self.identifier, k=self.param_dict['stride'],
                M=self.param_dict['w_size'], shuffle=False, verbose=0, n=None)
            datasets_dict[dataset_name] = data
        self.datasets_dict = datasets_dict

    @timed
    def standardize_data(self):
        """ Normalize and center data based on the training set. """
        from sklearn.preprocessing import StandardScaler
        # Train the scaler based on training dataset
        data_train = self.datasets_dict['train'].copy()
        columns = data_train.columns
        y_train = data_train['fault'].copy().to_numpy()
        X_train = data_train.drop('fault', axis=1)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        # Standardize every dataset
        datasets_dict = {}
        for dataset, data in self.datasets_dict.items():
            y = data['fault'].copy().to_numpy()
            X = data.drop('fault', axis=1)
            X_scaled = scaler.transform(X)
            data_np = np.concatenate(
                [X_scaled, y.reshape((len(y), 1))], axis=1)
            datasets_dict[dataset] = pd.DataFrame(data_np, columns=columns)
        self.preprocessing = scaler
        self.datasets_dict = datasets_dict

    @timed
    def postprocess(self, plot=False, plot_freq=0.01):
        """
        Loop over the available datasets to calculate metrics

        Usage:

          > extract_labels_and_preds (implemented in model.NN)
          > calc_full_scores
            > calc_scores (overall scores)
            > identify_fault (get fault number from a specific scenario)
            > measure_delay
          > tabulate_scores

        """
        self.log_start_section()
        scores_df_list = []
        for dataset_name, filepaths in self.filepath_dict.items():
            logger.info(
                f'Calculating scores for {dataset_name} dataset...')

            # Get labels and y_scores
            y_true, y_true_classes, y_pred, y_scores, case_indices, \
                case_lengths = self.extract_labels_and_preds(filepaths,
                                                             dataset_name)

            # Calculate scores
            if self.savefig:
                fig_filename = f'roc_{dataset_name}_{self.model_name}.png'
                fig_path = os.path.join(self.dst_dir, fig_filename)
            else:
                fig_path = None
            scores_dict, accuracies_dict, delays_dict \
                = self.calc_full_scores(y_true, y_true_classes, y_pred,
                                        y_scores, case_indices, case_lengths,
                                        dataset_name,
                                        plot=plot, plot_freq=plot_freq,
                                        fig_path=fig_path)

            # Format results
            scores_df = self._tabulate_scores(scores_dict, accuracies_dict,
                                              delays_dict, dataset_name)
            scores_df_list.append(scores_df)
        scores = pd.concat(scores_df_list, axis=1)
        self.scores = scores
        return scores

    def calc_full_scores(self, y_true, y_true_classes, y_pred, y_scores,
                         case_indices, case_lengths, dataset_name, plot=False,
                         plot_freq=0.01, fig_path=None):
        """
        Method that calculates all the scores based on previously calculated
        label (y) true, predicted and scores values.

        The y_scores refer to the numerical results previous to assign to a
        specific class, and is obtained by a `decision_function` method in
        sklearn or `predict` in Keras.

        `case_indices` and `case_lengths` are arrays corresponding to the
        successive scenarios that form y_pred, y_true, ...
        """
        # Init
        delays_dict = dict.fromkeys(self.idv_list, [])
        accuracies_dict = {}
        scenarios_path = None
        if plot and self.savefig:
            # Create dir for scenario plots if not existing
            scenarios_path = os.path.join(self.dst_dir, 'scenarios')
            if not os.path.isdir(scenarios_path):
                os.mkdir(scenarios_path)

        # OVERALL scores
        scores_dict = calc_scores(y_true, y_pred, y_scores, self.scores_list,
                                  fig_path)

        # DELAYS case by case
        for i in range(len(case_lengths)):
            # Get slices delimiting the case
            idx_ini, idx_end = case_indices[i], case_indices[i + 1]
            my_slice = slice(idx_ini, idx_end)

            # Get idv number that identifies fault
            idv = identify_fault(y_true_classes[my_slice])

            # Plot only a subset of cases randomly
            flag_plot = False
            if plot and np.random.uniform() <= plot_freq:
                scenario_filename = f'{self.model_name}_{dataset_name}' \
                                    f'_slice_{idx_ini}_{idx_end}_IDV_{idv}.png'
                scenario_path = os.path.join(scenarios_path, scenario_filename)
                self.plot_scenarios(y_true[my_slice], y_pred[my_slice],
                                    y_scores[my_slice], scenario_path)
                flag_plot = True

            # Store delays for faulty data
            if np.any(y_true[my_slice]):
                # Get delay
                delay = self.measure_delay(y_true[my_slice], y_pred[my_slice])
                delays_dict[idv] = delays_dict[idv] + [delay]
                # Do not use delays_dict.append (assigns to all dict items)
                if flag_plot:
                    logger.info(
                        f'Analyzing delays in scenario of fault {idv}. '
                        f'Slice indices: are ({idx_ini}, {idx_end})')
                    logger.info(f'Delay is {delay}')
            else:
                if flag_plot:
                    logger.info(
                        f'Scenario of slice indices ({idx_ini}, {idx_end}) '
                        f'is NOC. No delays to analyze')

        # Perform the average of the delays per each class
        for idv, delay_list in delays_dict.items():
            # Check if delay_list is empty to avoid a numpy warning
            if delay_list:
                delays_dict[idv] = np.mean(delay_list)
            else:
                delays_dict[idv] = None

        # ACCURACY PER CLASS
        for idv in self.idv_list_noc:
            # Get the slice that corresponds to an instance of class idv
            my_slice = y_true_classes == idv
            if any(my_slice):
                accuracies_dict[idv] = accuracy_score(y_true[my_slice],
                                                      y_pred[my_slice])
                logger.info(
                    f'Accuracy for fault {idv} is {accuracies_dict[idv]}')
            else:
                logger.info(f'No instances found of class {idv}')
                accuracies_dict[idv] = None

        # Print
        logger.debug(f'delays_dict: {delays_dict}')
        logger.debug(f'accuracies_dict: {accuracies_dict}')
        for metric, result in scores_dict.items():
            logger.info(f'{metric}: {result}')

        return scores_dict, accuracies_dict, delays_dict

    def plot_scenarios(self, y_true, y_pred, y_scores, scenario_path):
        """
        Plot a given scenario using y_true, y_pred and y_scores.

        Regarding the calculation of the time vector, the value:
            (len(y_true) - 1) * timestep_h * shift + timestep_h
        should be the exact value of the last time step where a window starts.

        To make np.arange work, an extra step (+ timestep) is needed,
        otherwise the exact last time step is not generated.

        """
        # Plot init
        timestep_h = self.timestep / 3600  # Plot in hours
        shift = self.param_dict['shift']
        t_last = (len(y_true) - 1) * timestep_h * shift + 2 * timestep_h
        t = np.arange(
            timestep_h,         # start
            t_last,             # stop
            timestep_h * shift  # step
        )
        assert len(t) == len(y_true), ('"t" dimensions do not match '
                                       'with "y". Check the generation of '
                                       'the time vector with np.arange.')
        # Plot
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        ax1.plot(t, y_pred, 'bx', label='y pred')
        ax1.plot(t, y_true, '-', color='royalblue', label='y true')
        ax2.plot(t, y_scores, 'g-', label='y scores')
        if self.threshold is not None:
            ax2.plot(t, len(t) * [self.threshold], '--', color='limegreen',
                     label='threshold')
        # Plot formatting
        ax1.set_xlabel('Time (h)')
        ax1.set_ylabel('Fault occurrence')
        ax1.set_yticks([0, 1])
        ax1.grid(True)
        ax1.legend(loc='upper left')
        ax2.set_ylabel('Score')
        ax2.legend(loc='upper right')
        if self.savefig:
            plt.savefig(scenario_path, dpi=300)
        else:
            plt.show()

    def measure_delay(self, y_true, y_pred):
        """
        Return the delay between the occurrence of a fault and its first
        detection. It calls itself recursively upon an early false positive.
        Return np.nan if detection is not accomplished and None if y_true was
        not a faulty case at all

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
        y_true : array-like
            true label values.
        y_pred : array-like
            predicted label values.

        """
        # Init
        steps2detect = self.steps2detect  # 6
        timestep = self.timestep  # 36

        # Get first occurrence of a fault both in true and predicted state
        start_index_true = where(y_true)
        if start_index_true is None:
            # Early return if the case is not truly faulty
            logger.info(
                'Running measure_delay() over non-faulty data. Returning None')
            return None

        # Check if the model predicts any fault at all, otherwise return NaN
        if np.any(y_pred):
            kernel = np.ones((steps2detect,))
            convolved_series = np.convolve(y_pred, kernel)
            try:
                start_index_pred = \
                    np.where(convolved_series == steps2detect)[0][0] \
                    - (steps2detect - 1)
            except IndexError:
                # if np.where does not return a tuple it will mean there is no
                # occurrence of the full sequence so no "formal" detection
                # appear in the prediction
                return np.nan
        else:
            return np.nan

        index_diff = start_index_pred - start_index_true

        if index_diff >= 0:
            # Return delay in minutes
            return index_diff * timestep / 60
        else:
            # Ignore early false positive and call this function recursively
            return self.measure_delay(y_true[start_index_pred + 1:],
                                      y_pred[start_index_pred + 1:])

    def _tabulate_scores(self, scores_dict, accuracies_dict, delays_dict,
                         dataset_name):
        # Accumulate results (WARNING: keep same order as in `metric_names`)
        # We could use dict.values(), but that does not make sure of the order
        results = [scores_dict[metric.__name__]
                   for metric in self.scores_list] \
                  + [accuracies_dict[idv] for idv in self.idv_list_noc] \
                  + [delays_dict[idv] for idv in self.idv_list]

        # Store results
        results_df = pd.DataFrame(results, index=self.metric_names,
                                  columns=[dataset_name])
        return results_df

    def extract_labels_and_preds(self, filepaths, dataset_name):
        return [], [], None, None, [], []

    def set_model_name(self):
        pass
