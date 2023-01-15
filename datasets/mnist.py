import matplotlib.pyplot as plt
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def plot_mnist(X, y, n=3, title=''):
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(10, 3))
    i = 0
    for ax, image, label in zip(axes, X, y):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
        i += 1
        if i > n:
            break
    fig.suptitle(title)


def get_mnist_5(split=False):
    # Load data
    digits = datasets.load_digits()

    # Get input and output data
    X = digits.data
    labels = digits.target
    y = labels == 5

    if split:
        # Split data into 50% train and 50% test subsets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True
        )
        return X_train, X_test, y_train, y_test
    else:
        return X, y


def load_toy_dataset_torch(batch_size):
    X_train, X_test, y_train, y_test = get_mnist_5(split=True)
    # Get only 5 images
    X_train = X_train[y_train == 1]  # 1 means occurrence of 5
    y_train = y_train[y_train == 1]
    # Transform to torch Dataloader objects
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    dataset_train = TensorDataset(X_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size)
    # Plot a few examples
    plot_mnist(X_train, y_train, title='train set')
    plot_mnist(X_test, y_test, title='test set')
    return dataloader_train, X_test, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_mnist_5(split=True)
    print('debug this line to watch the X and y object sizes')
