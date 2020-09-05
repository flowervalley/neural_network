import numpy as np


class Net:
    def __init__(self, structure):
        """Initialize components based on specified structure."""
        # weights[i] is the weight matrix from layer i-1 to layer i.
        self.weights = [None] + [
            np.random.randn(m, n) for n, m in zip(structure, structure[1:])
        ]
        self.biases = [np.random.rand(n, 1) for n in structure]

    def feedforward(self, input):
        """Feed input vector through the network and return ouput vector."""
        a = input
        for i in range(1, len(self.weights)):
            a = self.activation_function(self.weights[i] @ a + self.biases[i])
        return a

    def activation_function(self, z):
        """sigmoid function"""
        return 1 / (1 + np.exp(-z))

