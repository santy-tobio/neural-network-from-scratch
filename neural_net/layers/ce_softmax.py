import cupy as cp

from .base import Layer


class CESoftmax(Layer):
    """Combined Cross-Entropy Loss + Softmax activation

    This combines softmax activation with cross-entropy loss for numerical stability.
    The backward pass computes the gradient of CE loss w.r.t. logits.
    """

    def forward(self, logits: cp.ndarray):
        # Subtract max for numerical stability
        exp_input = cp.exp(logits - cp.max(logits, axis=0, keepdims=True))
        # Returns only softmax output
        self.output = exp_input / cp.sum(exp_input, axis=0, keepdims=True)
        return self.output

    def evaluate(self, logits: cp.ndarray) -> cp.ndarray:
        exp_input = cp.exp(logits - cp.max(logits, axis=0, keepdims=True))
        return exp_input / cp.sum(exp_input, axis=0, keepdims=True)

    def backward(self, prev_grad: cp.ndarray):
        # NOTE: This backward is the CrossEntropy + Softmax derivative (Not pure Softmax)
        return self.output - prev_grad
