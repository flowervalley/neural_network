import numpy as np
from mnist import MNIST

# MNIST data files are stored in '[project]/data/MNIST'.
_mndata = MNIST('../data/MNIST')

# helper variables
_label_map = [np.array([[int(i == j)] for i in range(10)]) for j in range(10)]
_reverse_vector = np.array([[i + 1] for i in range(10)])


def _mnist_to_numpy(images, labels):
    """Transform MNIST images and labels to numpy vector arrays."""
    image_vectors = np.array(images).reshape(len(images), 784, 1) / 255
    label_vectors = np.array([_label_map[x] for x in labels])
    return image_vectors, label_vectors


def load_training():
    """Return MNIST training data and labels as tuple."""
    return _mnist_to_numpy(*_mndata.load_training())


def load_test():
    """Return MNIST test data and labels as tuple."""
    return _mnist_to_numpy(*_mndata.load_testing())
