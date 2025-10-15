import numpy as np
import pandas as pd
import cupy as cp

# layer(input) -> output
# -
# -
# neural_net(layers,)
# - eval(input) -> output
# - loss(predicted, actual) -> loss

# Layer
# Linear(Layer)
# Relu(Layer)
# [Linear, Relu, Linear, Softmax]

# Layer
# Linear(Layer)
# Relu(Linear)
# [Relu, Relu, Softmax]

# LinearLayer
# Relu(LinearLayer)
# Softmax(LinearLayer)
# [Relu, Relu, Softmax]


class Layer:
    def forward(self, _):
        raise NotImplemented

    def backward(self, _):
        raise NotImplemented


class Linear(Layer):
    def __init__(self, input_dim: int, output_dim: int, batch_size: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        sigma = cp.sqrt(2.0 / (input_dim + output_dim))

        # cache para backward
        self.input = cp.zeros((input_dim, batch_size), dtype=cp.float32)
        self.grad_weights = cp.zeros((output_dim, input_dim), dtype=cp.float32)
        self.grad_bias = cp.zeros((output_dim, batch_size), dtype=cp.float32)

        # TODO: Merge into weight0 = bias
        self.weights = (
            cp.random.randn((output_dim, input_dim), dtype=cp.float32) * sigma
        )
        self.bias = cp.zeros((output_dim, batch_size), dtype=cp.float32)

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        # weight: (output_dim, input_dim) @ x: (input_dim, batch_size) -> (output_dim, batch_size)
        # bias: (output_dim, batch_size)
        # return: (output_dim, batch_size)
        self.input = x
        return self.weights @ x + self.bias

    def backward(self, prev_grad: cp.ndarray) -> cp.ndarray:
        # Bias
        self.grad_bias = cp.sum(prev_grad, axis=1, keepdims=True) / self.batch_size
        self.grad_weights = prev_grad @ self.input.T
        # Weights
        return self.grad_weights.T @ prev_grad


class NeuralNetwork:
    num_clases = 47
    hidden_layers = 0
    neurons_per_layer = 0
    batch_size = 3
    layers = []

    def __init__(self):
        self.layers = [Relu()]

    def forward():
        pass

    def backward():
        pass


class Relu(Layer):
    def forward(self, z_prev: cp.ndarray):
        return cp.max(0, self.weights * z_prev + self.bias)

    def backward(self, output):
        raise NotImplemented


class Softmax(Layer):
    def forward(self, input):
        return input
