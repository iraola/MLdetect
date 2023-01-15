import logging
import os

from tensorflow import keras

from .nn import NN, last_accuracy
from teutils import stringify_float

logger = logging.getLogger(__name__)


class RNN(NN):

    def __init__(self, prefix, identifier, drop_file, filepath_dict,
                 dst_dir=''):
        super().__init__(prefix, identifier, drop_file, filepath_dict, dst_dir)
        self.model_type = 'rnn'

    def set_model_name(self):
        logger.info('Setting a new model name...')

        # Init
        max_decimals = 5
        n_hidden = self.param_dict['n_hidden']
        n_neurons = self.param_dict['n_neurons']
        optimizer = self.param_dict['optimizer']
        w_size = self.param_dict['w_size']
        shift = self.param_dict['shift']
        stride = self.param_dict['stride']
        dropout = stringify_float(self.param_dict['dropout'], max_decimals)
        recurrent_dropout = stringify_float(
            self.param_dict['recurrent_dropout'], max_decimals)
        if isinstance(self.identifier, list):
            identifier = ''
        else:
            identifier = self.identifier

        # Set the model name
        model_name = self.prefix + '_' + self.model_type \
            + '_h' + str(n_hidden) \
            + '_n' + str(n_neurons) \
            + '_' + str(optimizer) + '_w' + str(w_size) \
            + '_sh' + str(shift) + '_st' + str(stride) \
            + '_d' + str(dropout) + '_rd' + str(recurrent_dropout) \
            + '_detect_' + identifier

        # Set a new checkpoint callback depending on the new model name
        dst_path_model = os.path.join(self.dst_dir, model_name + '.h5')
        checkpoint_cb = keras.callbacks.ModelCheckpoint(dst_path_model)
        self.callbacks = [checkpoint_cb]
        self.model_name = model_name
        return model_name

    def init_default(self, w_size, n_hidden):
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

        # Network architecture
        arch_param_dict = {
            'input_shape': (None, len(self.keep_index) - 1),
            'n_neurons': 128,
            'n_out': 1,  # Binary classifier
            'n_hidden': n_hidden,
            'activation': 'tanh',
            # 'initializer' : 'he_normal',
            'activation_out': 'sigmoid',
            # 'kernel_init_out' : 'glorot_uniform',
            # 'regularizer_value' : 1e-6,
            'dropout': 0.0,
            'recurrent_dropout': 0.0
        }
        # arch_param_dict['regularizer'] =
        #   l2(arch_param_dict['regularizer_value'])

        # Training parameters
        train_param_dict = {
            'batch_size': 32,
            'optimizer': 'adam',
            # Note that adam has also hyparams that can be tuned
            'loss_func': 'binary_crossentropy',
            'metrics_list': [last_accuracy],
            'custom_objects': {'last_accuracy': last_accuracy}
        }

        # Join dicts
        param_dict = {**data_param_dict, **arch_param_dict, **train_param_dict}
        self.param_dict = param_dict

        # Monitoring and others
        # # Obtain a list of the form, which is the output of fit's history:
        # #     ['loss', 'accuracy', 'val_loss', 'val_accuracy']
        aux_metrics_list = ['loss'] + self.param_dict['metrics_list']
        # If using custom metrics, some may need to be stringified
        for i, metric in enumerate(aux_metrics_list):
            if callable(metric):
                aux_metrics_list[i] = metric.__name__
        aux_metrics_list = aux_metrics_list \
            + ['val_' + metric for metric in aux_metrics_list]
        self.history_dict = dict.fromkeys(aux_metrics_list, [])

        # Compose model name
        self.set_model_name()
