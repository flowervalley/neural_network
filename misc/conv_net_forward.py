#%%
import numpy as np


class ConvLayer:
    def __init__(self, filters, biases):
        self.filters = filters
        self.biases = biases

    def forward(self, x):
        x_w, x_h, _ = x.shape
        n, f_w, f_h, _ = self.filters.shape

        w = x_w - f_w + 1
        h = x_h - f_h + 1

        y = np.zeros((w, h, n))

        for i in range(w):
            for j in range(h):
                y[i, j] = np.sum(
                    x[i : i + f_w, j : j + f_h] * self.filters, axis=(1, 2, 3)
                )

        y += self.biases

        return self.relu(y)

    def relu(self, x):
        return x * (x > 0)

    def __str__(self):
        return f'Conv {self.filters.shape[0]} {self.filters.shape[1:]}'

    def __repr__(self):
        return str(self)


class MaxPoolLayer:
    def __init__(self, size):
        self.size = size

    def forward(self, x):
        x_w, x_h, x_d = x.shape

        w = x_w // self.size
        h = x_h // self.size
        d = x_d

        y = np.zeros((w, h, d))

        for i in range(w):
            for j in range(h):
                y[i, j] = np.amax(x[i : i + self.size, j : j + self.size], axis=(0, 1))

        return y

    def __str__(self):
        return f'MaxPool 1/{self.size}'

    def __repr__(self):
        return str(self)


class FlatteningLayer:
    def forward(self, x):
        return x.reshape(np.prod(x.shape), 1)

    def __str__(self):
        return f'Flattening'

    def __repr__(self):
        return str(self)


class DenseLayer:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def forward(self, x):
        return self.relu(self.weights @ x + self.biases)

    def relu(self, x):
        return x * (x > 0)

    def __str__(self):
        return f'Dense {self.weights.shape[1]} -> {self.weights.shape[0]}'

    def __repr__(self):
        return str(self)


class SoftmaxLayer:
    def forward(self, x):
        print(x)
        e = np.exp(x)
        print(e)
        return e / e.sum()

    def __str__(self):
        return f'Softmax'

    def __repr__(self):
        return str(self)


class ConvNet:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x



net = ConvNet(
    [
        # 32 x 32 x 3
        ConvLayer(np.random.randn(8, 5, 5, 3), np.random.randn(8)),
        # 28 x 28 x 8
        MaxPoolLayer(2),
        # 14 x 14 x 8
        ConvLayer(np.random.randn(16, 5, 5, 8), np.random.randn(16)),
        # 10 x 10 x 16
        MaxPoolLayer(2),
        # 5 x 5 x 16
        ConvLayer(np.random.randn(32, 5, 5, 16), np.random.randn(32)),
        # 1 x 1 x 32
        FlatteningLayer(),
        # 32 x 1
        DenseLayer(np.random.randn(16, 32), np.random.randn(16, 1)),
        # 16 x 1
        DenseLayer(np.random.randn(16, 16), np.random.randn(16, 1)),
        # 16 x 1
        SoftmaxLayer(),
        # 16 x 1
    ]
)

x = np.random.rand(32, 32, 3)

# %%
