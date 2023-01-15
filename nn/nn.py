# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:40:06 2022

@author: Eduardo.Iraola
"""

# Basic
import logging
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tempfile

# Custom functions
from teutils import fetch_datasets, calc_n_windows_filelist, timed
from .base import Base

# Deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import InputLayer, Dense, LSTM, TimeDistributed
from tensorflow.keras.regularizers import l2

# Logger setup
logger = logging.getLogger(__name__)


# Custom metric
def last_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(y_true[:, -1], y_pred[:, -1])


class NN(Base):
    """
    API class to handle the training workflow of a keras neural network.

    Typical usage is:
        Initialize the object
        >> model_api = NN(prefix, identifier, drop_file, filepath_dict)

        Use default parameters to initialize the net
        >> model_api.init_default(w_size, n_hidden)
        And possibly custom some of them
        >> model_api.param_setter(n_neurons=10, n_hidden=4)

        Calculate mean and variance of the features
        >> model_api.calc_dataset_params()

        Collect data in tf.data format
        >> model_api.prepare_datasets()

        Build model
        >> model_api.build_model()

        Train
        >> model_api.keep_and_fit(valid_set_name='test', epochs=10)

        Get scores
        >> model_api.postprocess()

        Save model and extra_data
        >> model_api.save()

    """

    def __init__(self, prefix, identifier, drop_file, filepath_dict,
                 dst_dir=''):
        super().__init__(dst_dir)
        # Dummy init
        self.callbacks = None
        self.model_type = None
        self.datasets_dict = None
        self.keep_index = []
        # For model naming
        self.identifier = identifier
        self.prefix = prefix
        self._init_features(drop_file)
        self.filepath_dict = filepath_dict
        self.platform = 'tensorflow'
        self.flag_score = False
        # We do not store preprocess objects as in sklearn learners
        self.preprocessing = None
        self.scores = None
        self.param_dict = None

    def _init_features(self, drop_file):
        # Drop columns
        features = pd.read_csv(drop_file)
        keep_tags = features[features['drop'] == 0]['name'].to_list()
        drop_tags = features[features['drop'] == 1]['name'].to_list()
        keep_index = features[features['drop'] == 0]['index'].to_list()
        logger.debug('List of features kept for this model')
        for index, tag in zip(keep_index, keep_tags):
            logger.debug(f'{index} - {tag}')
        self.drop_tags = drop_tags
        self.keep_index = keep_index

    def param_setter(self, **kwargs):
        """
        Method to standardize the modification of default parameters of the
        neural network, specifically for `params_dict`.
        """

        if not kwargs:
            return None

        # Get list of parameter names
        param_list = list(kwargs.keys())

        for param_name in param_list:
            if param_name not in self.param_dict:
                logger.warning(
                    f'{param_name} did not exist in `param_dict`'
                    ' Make sure the spelling is correct.')
                logger.warning('Writing new parameter: ' + param_name)
            self.param_dict[param_name] = kwargs.pop(param_name)

        # Handle special cases
        if 'regularizer_value' in param_list:
            self.param_dict['regularizer'] = l2(
                self.param_dict['regularizer_value'])
        if 'layer_multiplier' in param_list:
            raise Exception('need to implement calc_neurons in param setter '
                            'if you want to change layer_multiplier')

        logger.info('Rebooting model name!')
        self.set_model_name()

        # Check remaining parameters passed
        if len(kwargs) != 0:
            raise ValueError('There are remaining parameters in the kwargs '
                             'dict that where not assigned:', param_list)

    def init_default(self, *args, **kwargs):
        """
        Initializer of the basic configuration of a neural net.

        To be extended through inheritance.

        """
        # Logger statement
        self.log_start_section()
        logger.info(
            'Initializing neural net parameters with default values...')

        if hasattr(self, 'param_dict') and self.param_dict:
            logger.warning('This instance already has a `param_dict` '
                           'attribute. Skipping parameters '
                           'initialization to avoid overwriting')
            return True
        else:
            return False

    @timed
    def calc_dataset_params(self):
        """
        Calculate the mean and standard deviation of the TRAIN SET for
        standardization. Also store the number of instances of the dataset,
        that will be necessary to account for the number of steps in each
        training epoch and is not equivalent to the number of resulting windows
        if w_size > 1.

        Use non-windowed data. If w_size is > 1, then expand the list of mean
        and std

        """
        self.log_start_section()
        logger.info(
            'Calculating mean and standard deviation of the training set...')

        # Init
        drop_tags = self.drop_tags
        identifier = self.identifier
        train_filepaths = self.filepath_dict['train']
        w_size = self.param_dict['w_size']
        shift = self.param_dict['shift']
        stride = self.param_dict['stride']

        # Fetch data (time-consuming operation)
        if 'X_mean' in self.param_dict \
                and 'X_std' in self.param_dict \
                and 'n_features' in self.param_dict:
            logger.warning('`X_mean`, `X_std` and `n_features` already exist'
                           ' in the params dict. Skipping their'
                           ' precalculation')
        else:
            # Use k=1 and M=1 on purpose, even though other config will apply
            df_full = fetch_datasets(train_filepaths, drop_list=drop_tags,
                                     identifier=identifier, k=1, M=1,
                                     shuffle=False, verbose=0)
            # Get features and account for Time column, which will be read
            # by LineTextDataset, and fault
            n_features = df_full.shape[1]
            # Calculate mean and standard deviation
            X_mean = df_full.drop(['fault'], axis=1).mean()
            X_std = df_full.drop(['fault'], axis=1).std()

            # Convert to windowed mean and std, model_type-dependent
            if self.model_type == 'dense':
                X_mean = np.array(X_mean.tolist() * w_size)
                X_std = np.array(X_std.tolist() * w_size)
            elif self.model_type == 'rnn':
                X_mean = np.array(X_mean)
                X_std = np.array(X_std)

            # Log and store values
            logger.info(f'X_mean array length: {len(X_mean)}')
            logger.debug(f'Example dataframe head {df_full.head()}')
            self.param_dict['n_features'] = n_features
            self.param_dict['X_mean'] = X_mean
            self.param_dict['X_std'] = X_std

        # Get number of instances of the training data
        n_instances = calc_n_windows_filelist(
            train_filepaths, w_size=w_size, shift=shift, stride=stride
        )

        # Logging
        logger.info('The total number of windows in the training set is '
                    f'{n_instances}')

        # Store
        self.param_dict['n_instances'] = n_instances

    def prepare_datasets(self, augment_data=False, augment_factor=10):
        """
        Generate tf.data.Dataset objects for each dataset so that they are
        prepared for training and evaluation.

        Allow the augmentation of the "train" dataset by multiplying negative
        instances

        """
        # Init
        filepath_dict = self.filepath_dict
        w_size = self.param_dict['w_size']
        shift = self.param_dict['shift']
        stride = self.param_dict['stride']

        self.log_start_section()
        logger.info(f'Preparing {list(filepath_dict.keys())} datasets '
                    f'as tf.data.Dataset objects...')

        # Loop over datasets
        datasets_dict = {}
        for dataset_name, filepaths in filepath_dict.items():
            if dataset_name == 'train':
                # The training set can be looped over and over
                repeat = None
                shuffle = True
                if augment_data:
                    filepaths = self._augment_data(filepaths, augment_factor)
            else:
                # The rest of sets exhaust after one epoch
                repeat = 1
                shuffle = False
            datasets_dict[dataset_name] = \
                self._csv_reader_dataset(filepaths,
                                         repeat=repeat,
                                         w_size=w_size,
                                         shuffle=shuffle,
                                         shift=shift,
                                         stride=stride,
                                         n_threads=None)
        self.datasets_dict = datasets_dict

        return datasets_dict

    def _augment_data(self, filepaths, factor=10):
        """
        Augment negative instances by a given factor.

        In the Rieth dataset, with 4.560.000 fault instances and 250.000
        + 190.000 NOC, instances a factor = 10 transforms it into:

        4.560.000 fault vs 2.500.000 + 190.000 NOC => 63 % fault vs. 27 % NOC

        """
        noc_filelist = [file for file in filepaths if 'IDV0_' in file]

        # Use (factor-1) since there is already a copy in original filepaths
        noc_filelist_augmented = (factor - 1) * noc_filelist
        augmented_filepath = filepaths + noc_filelist_augmented

        # Count the extra instances
        augmented_n_instances = calc_n_windows_filelist(
            augmented_filepath,
            w_size=self.param_dict['w_size'],
            shift=self.param_dict['shift'],
            stride=self.param_dict['stride']
        )

        logger.warning(
            f'Augmenting NOC train data by a factor of {factor}. '
            f'Previously: {self.param_dict["n_instances"]} windows, '
            f'Now: {augmented_n_instances} windows'
        )

        # IMPORTANT: Load the new number of instances (used for train steps
        # in `keep_and_fit`)
        self.param_dict['n_instances'] = augmented_n_instances

        return augmented_filepath

    def build_model(self):
        """
        Build and compile neural network model

        All parameters of the neural network should already be stored in the
        NN object

        """
        self.log_start_section()
        if hasattr(self, 'model') and self.model is not None:
            logger.warning('This instance already has a model, skipping '
                           'build model procedure')
            return

        logger.info('Building and compiling the neural network model...')

        # Init
        model_type = self.model_type
        # Network architecture
        input_shape = self.param_dict['input_shape']
        n_neurons = self.param_dict['n_neurons']
        n_out = self.param_dict['n_out']
        n_hidden = self.param_dict['n_hidden']
        activation = self.param_dict['activation']
        activation_out = self.param_dict['activation_out']
        # Regularization parameters
        if model_type == 'dense':
            regularizer = self.param_dict['regularizer']
        elif model_type == 'rnn':
            dropout = self.param_dict['dropout']
            recurrent_dropout = self.param_dict['recurrent_dropout']
        # Training parameters
        optimizer = self.param_dict['optimizer']
        loss_func = self.param_dict['loss_func']
        metrics_list = self.param_dict['metrics_list']

        # Initialize model
        model = keras.models.Sequential()

        # Input layer - input_shape should be [None, n_features - 1] for RNN
        model.add(InputLayer(input_shape=input_shape))

        # Build intermediate layers
        for layer in range(n_hidden):
            if model_type == 'dense':
                model.add(
                    Dense(n_neurons, activation=activation,
                          kernel_initializer=self.param_dict['initializer'],
                          kernel_regularizer=regularizer))
            elif model_type == 'rnn':
                model.add(
                    LSTM(n_neurons, return_sequences=True, dropout=dropout,
                         recurrent_dropout=recurrent_dropout))

        # Build output layer(s)
        if model_type == 'dense':
            model.add(
                Dense(n_out, activation=activation_out,
                      kernel_initializer=self.param_dict['kernel_init_out'],
                      kernel_regularizer=regularizer))
        elif model_type == 'rnn':
            model.add(TimeDistributed(Dense(n_out, activation=activation_out)))

        # Compile and checkpoints
        model.compile(loss=loss_func, optimizer=optimizer,
                      metrics=metrics_list)
        model.summary(print_fn=logger.info)

        self.model = model
        return model

    @timed
    def keep_and_fit(self, valid_set_name='test', epochs=30, steps=None,
                     verbose=1, ylim=None, plot=True):
        """
        Alternative version for keep_and_fit to use tf.data.Dataset objects.
        Train model and keep the previous history file. If it is the first run,
        the user must provide an empty history dict
        """
        self.log_start_section()
        logger.info('Training model...')

        # Init
        model = self.model
        train_set = self.datasets_dict['train']
        # Use logic `and` statement in case `valid_set_name` is None
        valid_set = valid_set_name and self.datasets_dict[valid_set_name]
        callbacks = self.callbacks
        history_dict = self.history_dict
        if steps is None:
            steps = self.param_dict['n_instances'] // self.param_dict[
                'batch_size']
            if steps == 0:
                logger.warning(
                    'steps = 0, probably due to a small dataset. '
                    'Artificially increasing steps to 1 to avoid a `fit` error'
                )
                steps = 1

        # Train model
        try:
            history_new = model.fit(train_set, steps_per_epoch=steps,
                                    epochs=epochs, validation_data=valid_set,
                                    callbacks=callbacks, verbose=verbose)
            # Add new history to old history dict
            for key, value in history_dict.items():
                # WARNING: throws an error if valid_set is not used in fit!
                history_dict[key] = value + history_new.history[key]
        except KeyboardInterrupt:
            logger.warning('\nTraining interrupted by the user.',
                           'The history dictionary will not be saved')

        # Plot evolution of training
        if plot:
            self.plot_training_curve(ylim)

        # Outputs
        logger.info(
            f'Trained for {len(history_dict["loss"])} epochs so far')

        # Ensure that the class attributes are updated
        self.model = model
        self.history_dict = history_dict
        return model

    def _csv_reader_dataset(self, filepaths, repeat=1,
                            shuffle_buffer_size=10000, batch_size=32, w_size=2,
                            shift=1, stride=1, shuffle=True, n_threads=None):
        """
        Whole data pipeline for tf.data treatment of 3D data composed by time
        window sequences

        Parameters
        ----------
        filepaths : list
            list of file ABSOLUTE paths of files with data to read
        repeat : int
            Repeats this dataset so each original value is seen 'repeat' times.
            Use '1' for validation and test sets and 'None' for unlimited
            repetition for training
        shuffle_buffer_size : int
            Must be around the size of the dataset for perfect shuffling

        Returns
        -------
        dataset : tf.data.Dataset

        """
        # Get dataset of files from actual list of paths (do not shuffle here)
        dataset = tf.data.Dataset.list_files(filepaths,
                                             shuffle=shuffle).repeat(repeat)
        # Map Dataset into TextLineDataset to start reading the file contents
        dataset = dataset.map(lambda filepath:
                              tf.data.TextLineDataset(filepath).skip(1),
                              num_parallel_calls=n_threads)
        # Flat map the list of TextLineDatasets and, at the same time,
        # map its contents into time windows 
        dataset = dataset.flat_map(lambda file_dataset:
                                   file_dataset.window(w_size,
                                                       drop_remainder=True,
                                                       shift=shift,
                                                       stride=stride))
        # Flat map again the results of window
        dataset = dataset.flat_map(lambda window: window.batch(w_size))
        # Shuffle
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size)
        # Apply preprocessing
        dataset = dataset.map(self._preprocess, num_parallel_calls=n_threads)
        # Batch
        dataset = dataset.batch(batch_size, num_parallel_calls=n_threads)
        # Prefetch
        return dataset.prefetch(1)

    def extract_labels_and_preds(self, filepaths, dataset_name):
        """
        Calculate the necessary arrays of values that are needed to input to
        a calc_full_scores method.
        """
        # Init
        w_size = self.param_dict['w_size']
        shift = self.param_dict['shift']
        stride = self.param_dict['stride']
        model = self.model

        # Load data from file
        data = self._csv_reader_dataset(filepaths, repeat=1, w_size=w_size,
                                        shuffle=False, shift=shift,
                                        stride=stride, n_threads=None)
        # Obtain scores and truth values
        y_scores = model.predict(data).squeeze()
        # `y_scores` is actually sklearn "scores" not a direct class prediction
        y_true = np.concatenate([y for x, y in data], axis=0).squeeze()

        # Take care of dimensionality
        if self.model_type == 'rnn':
            y_scores = y_scores[:, -1]

        # Calculation of `case_lengths` and `case_indices`
        # (length and indices that form each scenario in the dataset)
        case_lengths = []
        for filepath in filepaths:
            n_windows = calc_n_windows_filelist(
                [filepath],
                w_size=w_size,
                shift=shift,
                stride=stride,
                header=1
            )
            case_lengths.append(n_windows)
        case_indices = np.cumsum([0] + case_lengths)

        # Get actual fault classes (from 0 to 20)
        self.flag_score = True
        data_classes = self._csv_reader_dataset(
            filepaths, repeat=1, w_size=w_size, shuffle=False, shift=shift,
            stride=stride, n_threads=None
        )
        self.flag_score = False  # Important to turn down the flag
        y_true_classes = np.concatenate([y for x, y in data_classes],
                                        axis=0).squeeze()

        # Finally, get the predicted binary classes
        y_pred = y_scores.copy()
        y_pred = (y_pred >= 0.5).astype(int)

        if self.debug:
            plt.figure(figsize=(24, 6))
            plt.plot(y_true, label='y true')
            plt.plot(y_true_classes, 'r.', label='y true class')
            plt.plot(y_scores, label='y scores')
            plt.plot(y_pred, label='y pred')
            plt.legend(loc='upper left')
            plt.show()

        return y_true, y_true_classes, y_pred, y_scores, \
            case_indices, case_lengths

    def _preprocess(self, line):
        """
        TensorFlow function to manipulate text lines into tensor data in a 3D
        window fashion. Also, standardize the data and format labels
        """
        X_mean = self.param_dict['X_mean']
        X_std = self.param_dict['X_std']
        n_features = self.param_dict['n_features']
        w_size = self.param_dict['w_size']
        n_out = self.param_dict['n_out']

        # Assumes X_mean, X_std, n_features, keep_index precomputed GLOBALLY
        # Defaults for input data in case of empty instance
        defs = [0.] * n_features
        fields = tf.io.decode_csv(line, record_defaults=defs,
                                  select_cols=self.keep_index)
        # Stack scalar tensors into 1D tensors (necessary)
        x = tf.stack(fields[:-1])
        # Transpose it since stacking the window yields a shape (features,
        # timesteps) instead of (timesteps, features)
        x = tf.transpose(x)
        if self.model_type == 'dense':
            # n_features - n_out = 51 features (de las que una es 'fault') - 1
            x = tf.reshape(x, [(n_features - n_out) * w_size, ])
        # Detection labelling
        y = tf.stack(fields[-1:])
        y = tf.transpose(y)
        y = y[-1]
        # Set labels to 1 if different from 0
        if not self.flag_score:
            y = tf.where(tf.math.not_equal(y, 0), 1.0, y)
        return (x - X_mean) / X_std, y

    def __call__(self, w_size, n_hidden, dst_dir=''):
        """
        Agglutinate model parameters initialization, model building and
        dataset preparation.
        """
        # Initialize model parameters by default values
        self.init_default(w_size=w_size, n_hidden=n_hidden, dst_dir=dst_dir)
        # Calculate mean and variance of the features, that will be needed
        # later for dataset standardization
        self.calc_dataset_params()
        # Collect data in tf.data format
        self.prepare_datasets()
        # Build model
        self.build_model()

    def add_regularization(self, regularizer_value=1e-6):

        # Initialize regularizer
        regularizer = l2(regularizer_value)
        model = self.model

        if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
            logger.warning("Regularizer must be a subclass of "
                           "tf.keras.regularizers.Regularizer")
            return

        for layer in model.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)

        # When we change the layers attributes, the change only happens in
        # the model config file
        model_json = model.to_json()

        # Save the weights before reloading the model.
        tmp_weights_path = os.path.join(tempfile.gettempdir(),
                                        'tmp_weights.h5')
        model.save_weights(tmp_weights_path)

        # load the model from the config
        model = tf.keras.models.model_from_json(model_json)

        # Reload the model weights
        model.load_weights(tmp_weights_path, by_name=True)

        # Recompile model
        model.compile(
            loss=self.param_dict['loss_func'],
            optimizer=self.param_dict['optimizer'],
            metrics=self.param_dict['metrics_list']
        )

        self.param_dict['regularizer_value'] = regularizer_value
        self.param_dict['regularizer'] = regularizer
        self.model = model
        return model

    def plot_training_curve(self, ylim):
        pd.DataFrame(self.history_dict).plot(figsize=(10, 6))
        plt.ylim([None, ylim])
        plt.xlabel('Number of epochs')
        plt.grid(True)
        if self.savefig:
            fig_filename = f'epochs_{self.model_name}.png'
            fig_path = os.path.join(self.dst_dir, fig_filename)
            plt.savefig(fig_path, dpi=300)
        else:
            plt.show()

    def set_model_name(self):
        pass
