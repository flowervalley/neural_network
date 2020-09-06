import numpy as np


class Net:
    def __init__(self, structure):
        """Initialize components based on specified structure."""
        # weight matrices
        self.ws = [np.random.randn(m, n) for n, m in zip([0] + structure, structure)]
        # biases
        self.bs = [np.random.rand(n, 1) for n in structure]
        # activations
        self.ys = [np.zeros((n, 1)) for n in structure]
        # z values
        self.zs = [np.zeros((n, 1)) for n in structure]

    def feedforward(self, input):
        """Feed input vector through the network and return ouput vector."""
        self.ys[0] = input
        for i in range(1, len(self.ws)):
            self.zs[i] = self.ws[i] @ self.ys[i - 1] + self.bs[i]
            self.ys[i] = self.g(self.zs[i])
        return self.ys[-1]

    def g(self, z):
        """activation function (sigmoid)"""
        return 1 / (1 + np.exp(-z))
