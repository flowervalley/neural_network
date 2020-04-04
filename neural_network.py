import numpy as np


class NeuralNetwork:
    def __init__(self, layers, eta=0.01):
        self.layers = layers
        self.eta = eta

        self.weights = [np.ones((1, 1))] + [
            np.random.randn(r, c) / c ** 0.5 for c, r in zip(layers[:-1], layers[1:])
        ]
        self.biases = [np.zeros((1, 1))] + [np.random.randn(r, 1) for r in layers[1:]]

        self.activations = [np.zeros(b.shape) for b in self.biases]
        self.z_vectors = [np.zeros(b.shape) for b in self.biases]
        self.deltas = [np.zeros(b.shape) for b in self.biases]

    def guess(self, input):
        """Feed input vector through the network and return output vector."""
        a = input
        for i in range(1, len(self.layers)):
            a = self.activation_function(self.weights[i] @ a + self.biases[i])
        return a

    def train(self, input, target):

        weights_d = [np.zeros(w.shape) for w in self.weights]
        biases_d = [np.zeros(b.shape) for b in self.biases]

        ws, bs = self.backpropagate(input, target)

        for i in range(1, len(self.layers)):
            weights_d[i] += ws[i]
            biases_d[i] += bs[i]

        for i in range(1, len(self.layers)):
            self.weights[i] -= weights_d[i] * self.eta
            self.biases[i] -= biases_d[i] * self.eta

    def backpropagate(self, input, target):

        weights_d = [np.zeros(w.shape) for w in self.weights]
        biases_d = [np.zeros(b.shape) for b in self.biases]

        # Feed forward input vector and save activations and z values in every layer.
        self.activations[0] = input
        for i in range(1, len(self.layers)):
            self.z_vectors[i] = (
                self.weights[i] @ self.activations[i - 1] + self.biases[i]
            )
            self.activations[i] = self.activation_function(self.z_vectors[i])

        # Calculate deltas for the last layer.
        self.deltas[-1] = self.cost_function_derivative(
            self.activations[-1], target
        ) * self.activation_function_derivative(self.z_vectors[-1])

        # Calculate deltas for all preceding layers.
        for i in range(len(self.layers) - 2, 0, -1):
            self.deltas[i] = (
                self.weights[i + 1].transpose() @ self.deltas[i + 1]
            ) * self.activation_function_derivative(self.z_vectors[i])

        # Calculate weight- and bias derivatives.
        for i in range(1, len(self.layers)):
            weights_d[i] = self.deltas[i] @ self.activations[i - 1].transpose()
            biases_d[i] = self.deltas[i]

        return weights_d, biases_d

    def cost_function_derivative(self, output, target):
        """mean squared error derivative"""
        return output - target

    def activation_function(self, v):
        """sigmoid function"""
        return 1 / (1 + np.exp(-v))

    def activation_function_derivative(self, v):
        """derivative of sigmoid function"""
        return self.activation_function(v) * (1 - self.activation_function(v))
