import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from teutils import bisection, calc_n_windows_filelist, stringify_number, \
    timed
from .nn import NN

logger = logging.getLogger(__name__)


class Encoder(nn.Module):

    def __init__(self, input_dim, n_neurons):
        super(Encoder, self).__init__()
        self.model = nn.Sequential()
        # Loop through layers
        for i in range(len(n_neurons)):
            if i == 0:
                n_in = input_dim
            else:
                n_in = n_neurons[i - 1]
            n_out = n_neurons[i]
            self.model.add_module(f'enc{i + 1}', nn.Linear(n_in, n_out))
            self.model.add_module(f'enc{i + 1}_relu', nn.ReLU())

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):

    def __init__(self, latent_dim, n_neurons):
        super(Decoder, self).__init__()
        self.model = nn.Sequential()
        # Loop through layers
        for i in range(len(n_neurons)):
            if i == 0:
                n_in = latent_dim
            else:
                n_in = n_neurons[i - 1]
            n_out = n_neurons[i]
            self.model.add_module(f'dec{i + 1}', nn.Linear(n_in, n_out))
            self.model.add_module(f'dec{i + 1}_relu', nn.ReLU())

    def forward(self, z):
        return self.model(z)


class AEModel(nn.Module):
    """ Implement an autoencoder in PyTorch. """
    def __init__(self, input_dim, n_neurons):
        # input_dim = 50 for TE detection without windows
        super(AEModel, self).__init__()
        # Separate n_neuron list in two parts
        n_half_layers = len(n_neurons) // 2
        encoder_neurons = n_neurons[:n_half_layers]
        decoder_neurons = n_neurons[n_half_layers:]
        # Build encoder-decoder system
        self.encoder = Encoder(input_dim=input_dim,
                               n_neurons=encoder_neurons)
        self.decoder = Decoder(latent_dim=encoder_neurons[-1],
                               n_neurons=decoder_neurons)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class AE(NN):

    def __init__(self, prefix, identifier, drop_file, filepath_dict,
                 dst_dir=''):
        super().__init__(prefix, identifier, drop_file, filepath_dict, dst_dir)
        self.threshold = None
        self.optimizer = None
        self.loss_func = None
        self.model_type = 'ae'
        self.platform = 'torch'

    def set_model_name(self):
        logger.info('Setting a new model name...')

        # Init
        n_neurons = self.param_dict['n_neurons']
        activation = self.param_dict['activation']
        optimizer = self.param_dict['optimizer']
        w_size = self.param_dict['w_size']
        shift = self.param_dict['shift']
        stride = self.param_dict['stride']
        regularizer_value = self.param_dict['regularizer_value']
        layer_multiplier = self.param_dict['layer_multiplier']
        if isinstance(self.identifier, list):
            identifier = ''
        else:
            identifier = self.identifier

        # Set the model name
        # Format n_neurons since it is a list of int, not a string
        n_neurons_str = '_'.join(str(item) for item in n_neurons)
        model_name = self.prefix + '_' + self.model_type \
            + '_n' + str(n_neurons_str) + '_m' + str(layer_multiplier) \
            + '_' + str(activation) \
            + '_' + str(optimizer) + '_w' + str(w_size) \
            + '_sh' + str(shift) + '_st' + str(stride) \
            + '_r' + str(regularizer_value) \
            + '_detect_' + identifier

        # Set a new checkpoint callback depending on the new model name
        self.model_name = model_name
        return model_name

    def init_default(self, w_size, n_hidden=2, layer_multiplier=1):
        """
        Initializer of the basic configuration of a dense binary classifier
        """
        if super().init_default():
            return

        # Data shape
        data_param_dict = {
            'w_size': w_size,
            'shift': 1,
            'stride': 1,
        }

        input_shape, latent_dim, n_neurons = \
            self.calc_ae_neurons(n_hidden, w_size, layer_multiplier)

        # Network architecture
        arch_param_dict = {
            'input_shape': input_shape,
            'latent_dim': latent_dim,
            'layer_multiplier': layer_multiplier,
            'n_neurons': n_neurons,
            'n_out': None,  # Binary classifier
            'n_hidden': n_hidden,
            'activation': 'relu',  # 'elu' + 'he_normal' recommended by Géron
            # 'initializer': 'he_normal',
            'activation_out': '',
            'kernel_init_out': None,
            'regularizer_value': 1e-6,
            # 'dropout' : None
            # 'recurrent_dropout' : None
        }

        # Training parameters
        train_param_dict = {
            'batch_size': 32,
            'optimizer': 'adam',
            # Note that adam has also hyperparameters that can be tuned
            'loss_func': 'mse',
            'metrics_list': [], # ['f1_val'],
            'custom_objects': None
            # self.custom_objects = {'last_accuracy': self._last_accuracy}
        }

        # Join dicts
        param_dict = {**data_param_dict, **arch_param_dict, **train_param_dict}
        self.param_dict = param_dict

        # Monitoring and others
        # # Obtain a list of the form, which is the output of fit's history:
        # #     ['loss', 'accuracy', 'val_loss', 'val_accuracy']
        aux_metrics_list = ['loss'] + self.param_dict['metrics_list']
        aux_metrics_list = aux_metrics_list
        #    + ['val_' + metric for metric in aux_metrics_list]
        self.history_dict = dict.fromkeys(aux_metrics_list, [])

        # Compose model name
        self.set_model_name()

    def build_model(self):
        """
        Override NN.build_model() to build the net using an external PyTorch
        class.
        """
        # Check if we already have a model loaded
        self.log_start_section()
        if hasattr(self, 'model') and self.model is not None:
            logger.warning('This instance already has a model, skipping '
                           'build model procedure')
            return
        logger.info('Building neural network model...')
        # Init
        input_shape = self.param_dict['input_shape']
        n_neurons = self.param_dict['n_neurons']
        # Make sure n_neurons here is a list (not the case in other subclasses)
        assert isinstance(n_neurons, list)
        # Build model
        model = AEModel(input_dim=input_shape, n_neurons=n_neurons)
        self.model = model
        return model

    def setup_solver(self, loss_name, optim_name, **kwargs):
        """ Set up loss function and select optimizer. """
        self.log_start_section()
        logger.info('Setting up solver...')
        # Set up loss function
        if loss_name == 'mse':
            loss_func = nn.MSELoss()
        else:
            loss_func = None
            logger.error(f'Loss function {loss_name} not implemented.')
        # Set up optimizer
        if optim_name == 'adam':
            optimizer = optim.Adam(self.model.parameters())
        else:
            optimizer = None
            logger.error(f'Optimizer {optim_name} not implemented.')
        # Save object
        self.loss_func = loss_func
        self.optimizer = optimizer

    @timed
    def keep_and_fit(self, valid_set_name=None, epochs=30, steps=None,
                     verbose=0, ylim=None, plot=True, debug=False):
        """
        Override NN.keep_and_fit to train a PyTorch autoencoder model instead
        of TensorFlow.
        """
        self.log_start_section()
        logger.info('Training model...')
        # Transform train set to torch Dataloader objects
        data_train = self.datasets_dict['train'].copy()
        if 'fault' in data_train.columns:
            data_train.drop('fault', axis=1, inplace=True)
        X_train = data_train.to_numpy()
        X_train = torch.Tensor(X_train)
        dataset_train = TensorDataset(X_train)
        dataloader_train = DataLoader(
            dataset_train, batch_size=self.param_dict['batch_size'])
        # Transform val set to torch Dataloader objects
        example_np = None
        X_val = None
        if valid_set_name:
            data_val = self.datasets_dict[valid_set_name].copy()
            if 'fault' in data_val.columns:
                data_val.drop('fault', axis=1, inplace=True)
            X_val = data_val.to_numpy()
            X_val = torch.Tensor(X_val)
            dataset_val = TensorDataset(X_val)
            dataloader_test = DataLoader(
                dataset_val, batch_size=self.param_dict['batch_size'])

        # Prepare example in debug mode
        if debug and valid_set_name:
            example_np = \
                np.concatenate((X_train[0].numpy(),
                                X_val[0].numpy())).reshape(2, 64)

        # TRAINING LOOP
        train_losses = []
        valid_metrics = []
        for epoch in range(1, epochs + 1):
            # Run epoch training
            running_loss, valid_metric = \
                self.run_epoch(dataloader_train, epoch,
                               valid_set_name=valid_set_name)
            train_losses.append(running_loss)
            if valid_set_name:
                valid_metrics.append(valid_metric)
            # Debug for mnist plotting
            if debug and valid_set_name:
                if epoch % (epochs / 5) == 0:
                    self.debug_mnist(epoch, example_np)
        # Update history dictionary of epochs
        for key, value in self.history_dict.items():
            if key == 'loss':  # TODO: take care of other cases besides train
                self.history_dict[key] = value + train_losses
            if key == 'f1_val':
                self.history_dict[key] = value + valid_metrics
        logger.info(
            f'Trained for {len(self.history_dict["loss"])} epochs so far')
        # Plot epochs
        if plot:
            self.plot_training_curve(ylim)
        # Calculate threshold
        self.calc_threshold(target_acc=0.99)

    def run_epoch(self, dataloader_train, epoch, valid_set_name):
        """ Run one epoch. Include epoch timing. """
        running_loss = 0
        t1 = time.perf_counter()
        i = 0
        for i, batch in enumerate(dataloader_train):
            # Get data from batch tuple
            inputs = batch[0]
            # Zero the optimizer gradients
            self.optimizer.zero_grad()
            # forward + backward + optimize pass
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, inputs)
            loss.backward()
            self.optimizer.step()
            # Account for loss
            running_loss += loss.item()
            # Debug print statement per batch
            logger.debug(f'Epoch {epoch}: batch number {i}')
        t2 = time.perf_counter()
        elapsed_time = stringify_number(t2 - t1)
        # Validation score per epoch
        if valid_set_name:
            valid_metric = self.eval_valid_set(valid_set_name)
            val_string_line = f', val_metric: {valid_metric:.3f}'
        else:
            valid_metric = None
            val_string_line = ''
        # Complete epoch loss accounting
        running_loss /= i + 1  # average loss through all batches
        # Epoch prompt
        logger.info(f'Epoch {epoch} ran {i + 1} batches '
                    f'- loss: {running_loss:.4f}{val_string_line} '
                    f'- ETA: {elapsed_time} s')
        return running_loss, valid_metric

    @timed
    def eval_valid_set(self, valid_set_name):
        """ Evaluation of a validation set over a specific metric. """
        if 'f1_val' in self.history_dict.keys():
            self.calc_threshold(target_acc=0.99)
            y_pred, _ = self.predict(self.datasets_dict[valid_set_name])
            y_true = self.datasets_dict[valid_set_name]['fault'].copy()
            # make sure y_true is binary
            y_true[y_true > 0] = 1
            # calc metric
            metric_value = f1_score(y_true, y_pred)
        else:
            metric_value = None
        return metric_value

    @timed
    def calc_threshold(self, target_acc=0.99):
        """
        Calculate detection threshold based on the desired target accuracy in
        the training dataset.
        """
        def eval_dataset(threshold):
            """ Get the accuracy of the train dataset for a given threshold """
            y_pred, _ = self.predict(self.datasets_dict['train'], threshold)
            y_true = self.datasets_dict['train']['fault']
            accuracy = accuracy_score(y_true, y_pred)
            return accuracy

        # Get an initial point
        x_ini = self.datasets_dict['train'].iloc[0]
        threshold_ini = self.decision_function(x_ini)[0]
        # Apply bisection to get the desired threshold
        threshold = bisection(
            x_ini=threshold_ini,
            y_target=target_acc,
            func=eval_dataset,
            max_iter=1000,
            tol=1e-4
        )
        self.threshold = threshold
        return threshold

    def decision_function(self, X):
        """
        Predict confidence scores for the sample(s) in X. The scoring metric,
        the norm of the difference, is actually similar or the same as the MSE
        metric AND the Q score, as opposed to the T2 score.
        """
        # Transform data if necessary
        X = self.adapt_data_type(X)
        X_transformed = self.model(X)
        # Use `detach` to let the use of numpy functions in a Tensor later
        score = np.linalg.norm(X_transformed.detach() - X, axis=1)
        return score

    def predict(self, X, threshold=None):
        """
        Predict binary classification result of X based on decision_function.
        """
        y_score = self.decision_function(X)
        if threshold:
            y_pred = y_score > threshold
        else:
            y_pred = y_score > self.threshold
        return y_pred.astype(int), y_score

    def extract_labels_and_preds(self, filepaths, dataset_name):
        """
        Calculate the necessary arrays of values that are needed to input to
        a calc_full_scores method.

        Since this class currently pre-loads the whole datasets in
        self.datasets_dict, we don't reload each path as the NN class does
        to be able to cope with the TensorFlow dataset style.

        """
        # Init
        w_size = self.param_dict['w_size']
        shift = self.param_dict['shift']
        stride = self.param_dict['stride']

        # Get true values and predictions
        data = self.datasets_dict[dataset_name]
        y_true_classes = data['fault'].to_numpy().copy()
        y_true = y_true_classes > 0
        y_pred, y_score = self.predict(data)

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
        assert sum(case_lengths) == len(data), \
            'Mismatch between n_windows calculation and length of ' \
            'self.datasets_dict data'

        if self.debug:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax2 = ax1.twinx()
            ax2.plot([0, len(y_true)], [self.threshold, self.threshold], 'g--',
                     label='threshold')
            ax1.plot(y_true, 'b-', label='y true')
            ax1.plot(y_true_classes, 'b.', label='y true class')
            ax2.plot(y_score, 'g-', label='y scores')
            ax1.plot(y_pred, 'r-', label='y pred')
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax1.set_ylabel('Prediction', color='b')
            ax2.set_ylabel('Score', color='g')
            fig.suptitle(f'{dataset_name} dataset. '
                         f'Threshold = {self.threshold:.2f}')
            plt.show()

        return y_true, y_true_classes, y_pred, y_score, \
            case_indices, case_lengths

    @staticmethod
    def adapt_data_type(data):
        """ Make data suitable as input to the neural net model"""
        # Check data type
        if 'pandas' in str(type(data)):
            # Convert series in dataframe
            if 'series' in str(type(data)):
                data = pd.DataFrame(data).transpose()
            # Remove fault column if it exists
            if 'fault' in data.columns:
                data = data.drop('fault', axis=1)
            # If pandas, first move to numpy, then to torch Tensor
            data = data.to_numpy()
        else:
            raise Exception('Not implemented')
        # Check shape and convert to two-dimensional
        if len(data.shape) == 1:
            data = data.reshape((1, data.shape[0]))
        # Convert to pytorch tensor
        data = torch.Tensor(data)
        return data

    def calc_ae_neurons(self, n_hidden, w_size, layer_multiplier):
        """
        AE parameter setup based on 4 inputs:

        Arguments
        ---------
            n_hidden: number of hidden layers for each encoder and decoder
                sub-models.
            w_size: (to later be considered stride as well)
            layer_multiplier: a multiplier for decreasing number of neurons in
                each layer of the autoencoder.
        """
        # Calculation of input_shape
        input_shape = (len(self.keep_index) - 1) * w_size
        # Calculation of n_neurons
        mult = 1 + layer_multiplier / n_hidden
        n_neurons = (2 * n_hidden + 1) * [None]
        n_neuron = input_shape
        for i in range(n_hidden + 1):
            n_neurons[i] = n_neuron
            n_neurons[-(1 + i)] = n_neuron
            n_neuron = int(n_neuron / mult)
        latent_dim = n_neurons[n_hidden]
        # This n_neurons has the form [input_dim,...,latent_dim,..., input_dim]
        #  but we need to remove the first input_shape to arrange this list for
        #  the pytorch model
        n_neurons.pop(0)
        return input_shape, latent_dim, n_neurons

    def debug_mnist(self, epoch, example_np):
        example_tensor = torch.Tensor(example_np)
        pred_example = \
            self.model(example_tensor).round().detach().numpy()
        from models.datasets import plot_mnist
        plot_mnist(np.concatenate([example_np, pred_example]),
                   [0, 1, 0, 1], n=4, title=f'Epoch {epoch}')
