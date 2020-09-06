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
        """
        Feed input vector through the network and return output vector.
        Additionally store the activations and z values.
        """
        self.ys[0] = input
        for i in range(1, len(self.ws)):
            self.zs[i] = self.ws[i] @ self.ys[i - 1] + self.bs[i]
            self.ys[i] = self.g(self.zs[i])
        return self.ys[-1]

    def train(self, input, target):
        """
        Train the network based on a single input and target by computing 
        partial derivatives of the cost function with respect to weights 
        and biases using backpropagation.
        """
        # Feed the input through the net to get the activations and z values.
        self.feedforward(input)

        # Calculate partial derivatives of the cost dunction for all z values in all layers.
        ds = [0] * len(self.bs)
        ds[-1] = self.c_d(self.ys[-1], target) * self.g_d(self.zs[-1])
        for i in range(len(self.bs) - 2, 0, -1):
            ds[i] = (self.ws[i + 1].transpose() @ ds[i + 1]) * self.g_d(self.zs[i])

        # Update weights and biases based on partial derivatives.
        for i in range(1, len(self.ws)):
            self.ws[i] -= 0.1 * (ds[i] * self.ys[i - 1].transpose())
            self.bs[i] -= 0.1 * ds[i]

    def c(self, y, t):
        """cost function for actual y and expected t (MSE)"""
        return 1 / 2 * (np.sum((y - t) ** 2) ** 0.5) ** 2

    def c_d(self, y, t):
        """cost function derivative(MSE')"""
        return y - t

    def g(self, z):
        """activation function (sigmoid)"""
        return 1 / (1 + np.exp(-z))

    def g_d(self, z):
        """activation function derivative (sigmoid')"""
        a = self.g(z)
        return a * (1 - a)
