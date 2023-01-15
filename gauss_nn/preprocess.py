import logging
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from models.pca import PCA_trainer
from teutils import calc_n_windows, fetch_datasets

logger = logging.getLogger(__name__)


def plot_fit(order, thetas, timestep, window, x):
    n = len(window)
    x_fit = np.arange(0, n * timestep, timestep / 10)  # finer mesh
    y_fit = np.zeros(len(x_fit))
    for i in range(x_fit.shape[0]):
        pow = order
        for theta in thetas:
            y_fit[i] += theta * x_fit[i] ** pow
            # Decrease power for the next fitted parameter
            pow -= 1
    plt.figure()
    plt.plot(x, window, 'x')
    plt.plot(x_fit, y_fit)


class Preprocess:

    def __init__(self, model_name, alpha, w_size, stride, shift=1, order=2,
                 sample_time=36, statistics=('t2',)):
        self.times = None
        self.n_components = None
        self.scaler = None
        self.pca = None
        self.output_drop_tags = None
        self.input_tags = None
        self.output_tags = None
        self.statistics = statistics
        self.model_name = model_name
        self.alpha = alpha
        self.w_size = w_size
        self.stride = stride
        self.shift = shift
        self.order = order
        self.sample_time = sample_time

    def init_features(self, drop_file, debug=False):
        """ Features used for regression: input and output. """
        # Drop columns
        features = pd.read_csv(drop_file)
        input_tags = features[features['input'] == 1]['name'].to_list()
        output_tags = features[features['output'] == 1]['name'].to_list()
        output_drop_tags = features[features['output'] == 0]['name'].to_list()
        if debug:
            print('List of features kept for this model')
            for tag in input_tags + output_tags:
                print(f'{tag}')
        # Get number of output features (XMEAS) including windowization
        self.n_components = len(output_tags) * self.w_size
        # Store features
        self.input_tags = input_tags
        self.output_tags = output_tags
        self.output_drop_tags = output_drop_tags
        return input_tags, output_tags, output_drop_tags

    def preprocess_pca(self, noc_filepaths, debug=False):
        """
        Do preprocessing to initialize and fit the PCA object.

        Note: The PCA reduction is not important, but we have the statistic
        generator programmed in the PCA_detect class.

        """
        pca_api = PCA_trainer(
            w_size=self.w_size, train_filepaths=noc_filepaths,
            drop_tags=self.output_drop_tags, model_name=self.model_name,
            sample_time=self.sample_time, alpha=self.alpha,
            n_components=self.n_components, statistics=self.statistics,
            debug=debug
        )
        pca_api.preprocess()
        pca_api.fit()
        # Save to Preprocess object
        self.scaler = pca_api.scaler
        self.pca = pca_api.model
        return self.pca

    def characterize_window(self, window, timestep, plot=False):
        """
        Fit window to a specific `n` order polynomial and return coefficients.
        Assuming only 1 feature.
        """
        n = len(window)  # Number of samples in the window
        x = np.arange(0, n * timestep, timestep)
        thetas = np.polyfit(x, window, self.order)
        if plot:
            plot_fit(self.order, thetas, timestep, window, x)
        return thetas

    def characterized_input_df(self, filepath, inout_delay, debug=False):
        """
        Compose a dataframe characterizing the source dataframe with polynomial
        fits of the given `order`. Applies to the input_tags.

                XMEAS(1)                     XMEAS(2)
        theta1 | theta2 | theta 3 || theta1 | theta2 | theta3  ...

        """
        # Initializations
        w_size = self.w_size
        stride = self.stride
        shift = self.shift
        order = self.order
        input_tags = self.input_tags
        # Load data
        df = pd.read_csv(filepath, index_col='Time')
        n = len(df)
        n_windows = calc_n_windows(n, w_size=w_size, shift=shift,
                                   stride=stride)
        # Correction for this special case, calc_n_windows does not account for
        # TODO: n_windows skips some windows if shift > 1 (not too important)
        n_windows -= inout_delay

        # Time array setup
        if df.index.name == 'Time':
            # Times are in the index of the dataframe
            timestep = df.index[1] - df.index[0]
            time_start = df.index[0]
        else:
            # Probably times are in a normal feature column
            raise Exception('Not implemented')
        times = np.arange(
            time_start, time_start + n_windows * timestep * shift,
            timestep * shift)
        # Translate times to use the latest of the window to index the instance
        #  NOTE: Do not add inout_delay, since time must point to the input
        #        last time, not the output, which is located in the future
        times += ((w_size - 1) * stride) * timestep
        assert len(times) == n_windows, 'The number of windows does not match'
        self.times = times

        # Loop over the columns of the original dataframe
        feature_df_list = []
        for feature in df.columns:
            # Initialize mini-dataframe data
            feature_names = [f'{feature}_theta{i}'
                             for i in range(order, -1, -1)]
            feature_df_orig = df[feature]
            feature_array = np.zeros((n_windows, order + 1))
            # Loop through different windows in the given column
            for i, orig_i in enumerate(range(0, n, shift)):
                # `i_orig` takes into account the possible use of `shift`
                # `i` is the plain index from 1 to n_windows (or less)
                # TODO: Setup for loop better (deterministic) instead of break
                if i >= n_windows:
                    break
                # Shift window measurement if shift is different from 1
                #  convention is start:stop:step
                # Features (input) processing
                if feature in input_tags and not feature_df_orig.hasnans:
                    logger.debug(
                        f'characterized_input_df.Feature {feature} - Loop {i}')
                    window_features = \
                        feature_df_orig[orig_i:orig_i + w_size * stride:stride]
                    feature_array[i, :] = self.characterize_window(
                        window=window_features, timestep=timestep, plot=debug)

            # # Labels (output) processing
            # # NOTE: We don't do label processing here anymore
            # if feature in output_tags:
            #     window_labels = \
            #         feature_df_orig[orig_i + inout_delay:
            #                         orig_i + inout_delay + w_size * stride:
            #                         stride]
            #     label_array[i, :] = window_labels.mean(), window_labels.std()

            # Blend into a mini-dataframe for one feature
            feature_df = pd.DataFrame(feature_array, index=times,
                                      columns=feature_names)
            feature_df.index.name = 'Time'
            if feature in input_tags:
                feature_df_list.append(feature_df)
        # Concatenate resulting mini-dataframes
        dst_feature_df = pd.concat(feature_df_list, axis=1)
        return dst_feature_df

    def characterized_output_df(self, filepath, inout_delay):
        """
        Returns a dataframe with the output Hotelling's T2 statistic of a
        series of time windows.
        """
        X = fetch_datasets([filepath], drop_list=self.output_drop_tags,
                           identifier='', k=self.stride, M=self.w_size,
                           shuffle=False, verbose=0, n=None)
        X = X[inout_delay:]
        labels_df = X.pop('fault')  # `y` not needed
        X_scaled = self.scaler.transform(X)
        t2 = self.pca.decision_function(X_scaled, stat='t2')
        # Set up dataframe
        stat_df = pd.DataFrame(t2, index=self.times, columns=['T2'])
        stat_df.index.name = 'Time'
        labels_df.index = self.times
        labels_df.index.name = 'Time'
        output_df = pd.concat([stat_df, labels_df], axis=1)
        return output_df

    def prepare_datasets(self, filepath_dict, dst_dir, inout_delay,
                         debug=False):
        """
        Preprocess and compose dataframes for gaussian nn process.
        """
        # Save threshold
        th_path = os.path.join(dst_dir, 'threshold.csv')
        pd.DataFrame([self.pca.t2_th]).to_csv(th_path)
        # Loop over datasets
        for dataset_name, filepaths in filepath_dict.items():
            logger.info(f'Preparing files for {dataset_name} dataset.')
            for filepath in filepaths:
                # Get file name
                file_name = filepath.split(os.sep)[-1]
                dst_path = os.path.join(dst_dir, dataset_name, file_name)
                if os.path.isfile(dst_path):
                    logger.info(f'File {file_name} in dataset `{dataset_name}`'
                                f' already exists. Skipping file...')
                else:
                    logger.info(f'Processing file {file_name}')
                    # Load data and preprocess it
                    input_df = self.characterized_input_df(
                        filepath, inout_delay, debug)
                    output_df = self.characterized_output_df(
                        filepath, inout_delay)
                    # Concatenate input features and output features
                    new_df = pd.concat([input_df, output_df], axis=1)
                    # Write to file
                    new_df.to_csv(dst_path)


def main():
    # Handle logging
    import logging
    logging.basicConfig(level=logging.INFO)
    # Main parameters
    debug = True
    alpha = 0.01
    w_size = 25
    stride = 1
    shift = 1
    order = 2
    sample_time = 36
    inout_delay = 0  # I think that we shouldn't use a value higher than 0
    #                  since fundamentally that would imply detecting a fault
    #                  occurring in the future
    variant_extension = f'w{w_size}_st{stride}_sh{shift}'
    # File management variables
    src_dir_name = '01.NOC_only_residuals_SS'
    src_dir = os.path.join('..', '..', 'data', src_dir_name)
    dst_dir = os.path.join(src_dir, variant_extension)
    assert os.path.isdir(src_dir)
    assert os.path.isdir(dst_dir)
    dataset_names = ['train', 'train-dev', 'val', 'test']
    drop_file = os.path.join(src_dir, 'features_detect_gauss_nn.csv')
    # Dataset paths dictionary
    filepath_dict = {}
    for dataset in dataset_names:
        data_dir = os.path.join(src_dir, dataset)
        filepaths = [os.path.join(data_dir, file) for file in
                     os.listdir(data_dir) if file.endswith('.csv')]
        filepath_dict[dataset] = filepaths

    # Initialize Preprocess
    preprocess = Preprocess(
        model_name=src_dir_name.split('.')[1], alpha=alpha, w_size=w_size,
        stride=stride, shift=shift, order=order, sample_time=sample_time,
    )
    # Get usable tags
    preprocess.init_features(drop_file)
    # Launch PCA class to later produce Hotelling statistics
    preprocess.preprocess_pca(noc_filepaths=filepath_dict['train'],
                              debug=debug)
    # Prepare datasets
    preprocess.prepare_datasets(filepath_dict, inout_delay=inout_delay,
                                dst_dir=dst_dir)


if __name__ == '__main__':
    main()
