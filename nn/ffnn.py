import logging
import os

from tensorflow.keras.regularizers import l2
from tensorflow import keras

from models.nn import NN

logger = logging.getLogger(__name__)


class FFNN(NN):

    def __init__(self, prefix, identifier, drop_file, filepath_dict,
                 dst_dir=''):
        super().__init__(prefix, identifier, drop_file, filepath_dict, dst_dir)
        self.model_type = 'dense'

    def set_model_name(self):
        logger.info('Setting a new model name...')

        # Init
        n_hidden = self.param_dict['n_hidden']
        n_neurons = self.param_dict['n_neurons']
        activation = self.param_dict['activation']
        optimizer = self.param_dict['optimizer']
        w_size = self.param_dict['w_size']
        shift = self.param_dict['shift']
        stride = self.param_dict['stride']
        regularizer_value = self.param_dict['regularizer_value']
        if isinstance(self.identifier, list):
            identifier = ''
        else:
            identifier = self.identifier

        # Set the model name
        model_name = self.prefix + '_' + self.model_type + '_' \
            + 'h' + str(n_hidden) \
            + '_n' + str(n_neurons) + '_' + str(activation) \
            + '_' + str(optimizer) + '_w' + str(w_size) \
            + '_sh' + str(shift) + '_st' + str(stride) \
            + '_r' + str(regularizer_value) \
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
            'input_shape': ((len(self.keep_index) - 1) * w_size,),
            'n_neurons': 128,
            'n_out': 1,  # Binary classifier
            'n_hidden': n_hidden,
            'activation': 'elu',  # 'elu' + 'he_normal' recommended by Géron
            'initializer': 'he_normal',
            'activation_out': 'sigmoid',
            'kernel_init_out': 'glorot_uniform',
            'regularizer_value': 1e-6,
            # 'dropout' : None
            # 'recurrent_dropout' : None
        }
        arch_param_dict['regularizer'] = l2(
            arch_param_dict['regularizer_value'])

        # Training parameters
        train_param_dict = {
            'batch_size': 32,
            'optimizer': 'adam',
            # Note that adam has also hyparams that can be tuned
            'loss_func': 'binary_crossentropy',
            'metrics_list': ['accuracy'],
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
        aux_metrics_list = aux_metrics_list \
            + ['val_' + metric for metric in aux_metrics_list]
        self.history_dict = dict.fromkeys(aux_metrics_list, [])

        # Compose model name
        self.set_model_name()
