# import numpy as cp
import cupy as cp


class MLP:
    def __init__(
        self,
        batch_size: int,
        input_dim: int,
        output_classes: int,
        hidden_layers_neuron_count: list[int],
    ):
        # Architecture parameters
        self.batch_size = batch_size
        self.layers = []
        # [input, ..., hidden_layers] -> then [hidden_layer[-1], output_classes]
        self.relu_layers = [input_dim] + hidden_layers_neuron_count
        # Append all relu layers
        for layer in range(0, (len(self.relu_layers) - 1)):
            self.layers += [
                Linear(
                    self.relu_layers[layer],
                    self.relu_layers[layer + 1],
                    batch_size,
                ),
                Relu(),
            ]
        # Append last layer (Softmax)
        self.layers += [
            Linear(self.relu_layers[-1], output_classes, batch_size),
            CESoftmax(),
        ]

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y: cp.ndarray) -> cp.ndarray:
        prev_grad = y
        for layer in reversed(self.layers):
            prev_grad = layer.backward(prev_grad)
        return prev_grad

    def update_weights(self, learning_rate: float):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.weights -= learning_rate * layer.grad_weights
                layer.bias -= learning_rate * layer.grad_bias

    def loss(self, x: cp.ndarray) -> float:
        return cp.mean(-cp.log(cp.clip(x, 1e-10, 1.0)))

    def save_weights(self, weights_file: str):
        weights = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                weights.append(layer.weights.get())
                weights.append(layer.bias.get())
        cp.savez(weights_file, *weights)

    def load_weights(self, model, weights_file: str):
        weights = cp.load(weights_file)
        linear_layer_idx = 0
        for layer in model.layers:
            if isinstance(layer, Linear):
                w, b = (
                    weights[f"arr_{linear_layer_idx*2}"],
                    weights[f"arr_{linear_layer_idx*2+1}"],
                )
                layer.weights = cp.asarray(w)
                layer.bias = cp.asarray(b)
                linear_layer_idx += 1


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

        # TODO: Merge bias as Weight 0
        # He initialization for ReLU networks -> https://arxiv.org/pdf/1502.01852.pdf
        self.weights = cp.random.randn(output_dim, input_dim).astype(
            cp.float32
        ) * cp.sqrt(2.0 / input_dim)
        # Bias as a column vector [output_dim, 1] -> broadcasted during addition
        self.bias = cp.zeros((output_dim, 1), dtype=cp.float32)

        # cache para backward
        self.input = cp.zeros((input_dim, batch_size), dtype=cp.float32)
        self.grad_weights = cp.zeros((output_dim, input_dim), dtype=cp.float32)
        self.grad_bias = cp.zeros((output_dim, 1), dtype=cp.float32)

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        # weight: (output_dim, input_dim) @ x: (input_dim, batch_size) -> (output_dim, batch_size)
        # bias: (output_dim, batch_size)
        # return: (output_dim, batch_size)
        self.input = x
        return self.weights @ x + self.bias

    def backward(self, prev_grad: cp.ndarray) -> cp.ndarray:
        # prev_grad: (output_dim, batch_size), input: (input_dim, batch_size)
        # Weights: DL/DZ3 * (a^i-1)^T
        self.grad_weights = (
            prev_grad @ self.input.T
        ) / self.batch_size  # (output_dim, input_dim)
        # Bias gradient is the sum over the batch dimension -> (output_dim, 1)
        self.grad_bias = cp.sum(prev_grad, axis=1, keepdims=True) / self.batch_size

        # TODO: Why w.T * prev_grad and not w * prev_grad?
        return self.weights.T @ prev_grad


class Relu(Layer):
    def forward(self, x: cp.ndarray):
        self.mask = cp.array(x > 0, dtype=cp.float32)
        return x * self.mask

    def backward(self, prev_grad: cp.ndarray):
        return prev_grad * self.mask  # prev_grad @ diag(self.mask)


class CESoftmax(Layer):

    def forward(self, logits: cp.ndarray):
        # Subtract max for numerical stability
        exp_input = cp.exp(logits - cp.max(logits, axis=0, keepdims=True))
        # Returns only softmax output
        self.output = exp_input / cp.sum(exp_input, axis=0, keepdims=True)
        return self.output

    def backward(self, prev_grad: cp.ndarray):
        # TODO: Rightnow, the backward is the CrossEntropy + Softmax derivative (Not pure Softmax)
        return self.output - prev_grad
