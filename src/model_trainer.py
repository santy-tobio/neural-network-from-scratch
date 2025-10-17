import numpy as np
import pandas as pd
import cupy as cp


class ModelTrainer:
    def __init__(
        self,
        model,
        X_train: cp.ndarray,
        y_train: cp.ndarray,
        X_val: cp.ndarray,
        y_val: cp.ndarray,
        epochs: int,
    ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.epochs = epochs

    def train(self):
        for epoch in range(self.epochs):
            # Forward pass
            outputs = self.model.forward(self.X_train)
            # Compute loss
            loss = self.model.loss(outputs, self.y_train)
            # Backward pass
