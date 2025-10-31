import os

import cupy as cp
import numpy as np


class Dataset:
    """
    Handles data loading and splitting into train/val/test.
    This ensures ALL experiments use the same data splits,
    making comparisons fair.

    Memory-efficient: Loads and processes data in CPU (NumPy),
    then transfers only the final splits to GPU (CuPy).
    """

    def __init__(
        self,
        X_path: str,
        y_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        normalize: bool = True,
        stratify: bool = True,
        random_seed: int = 42,
    ):
        """
        Load and split dataset into train/val/test.
        Dataset is split with stratification to maintain class distribution.
        Features can be normalized to [0, 1].

        Args:
            X_path: Path to X data
            y_path: Path to y labels
            train_ratio: Proportion for training (default 0.7)
            val_ratio: Proportion for validation (default 0.1)
            test_ratio: Proportion for test (default 0.2)
            normalize: Whether to normalize features to [0, 1]
            stratify: Whether to maintain class distribution in splits
            random_seed: Random seed for reproducibility
        """
        total_ratio = train_ratio + val_ratio + test_ratio
        assert (
            abs(total_ratio - 1.0) < 1e-6
        ), f"Ratios must sum to 1.0, got {total_ratio}"
        assert (
            train_ratio > 0 and val_ratio >= 0 and test_ratio > 0
        ), "Ratios must be positive"
        assert os.path.exists(X_path) and os.path.exists(
            y_path
        ), "Data files must exist"

        print("Loading dataset in CPU (NumPy) to save GPU memory...")
        X_cpu = np.load(X_path)
        y_cpu = np.load(y_path)

        X_cpu = X_cpu.reshape(X_cpu.shape[0], -1)

        if normalize:
            X_cpu = X_cpu.astype(np.float32)
            max_val = X_cpu.max()
            X_cpu /= max_val

        print(
            f"Splitting dataset: {X_cpu.shape[0]} samples - "
            f"train: {train_ratio:.2f}, val: {val_ratio:.2f}, test: {test_ratio:.2f}"
        )

        if stratify:
            train_idx, val_idx, test_idx = self._stratified_split_3way_cpu(
                y_cpu, train_ratio, val_ratio, test_ratio, random_seed
            )
        else:
            train_idx, val_idx, test_idx = self._random_split_3way_cpu(
                X_cpu.shape[0], train_ratio, val_ratio, test_ratio, random_seed
            )

        X_train_cpu = X_cpu[train_idx]
        y_train_cpu = y_cpu[train_idx]
        X_val_cpu = X_cpu[val_idx]
        y_val_cpu = y_cpu[val_idx]
        X_test_cpu = X_cpu[test_idx]
        y_test_cpu = y_cpu[test_idx]

        del X_cpu, y_cpu

        print(
            "Keeping data in CPU (NumPy) - batches will be moved to GPU during training"
        )
        self.X_train = X_train_cpu
        self.y_train = y_train_cpu
        self.X_val = X_val_cpu
        self.y_val = y_val_cpu
        self.X_test = X_test_cpu
        self.y_test = y_test_cpu

        print("Dataset loaded:")
        print(f"  Train: {len(self.X_train)} samples")
        print(f"  Val:   {len(self.X_val)} samples")
        print(f"  Test:  {len(self.X_test)} samples")

    def _random_split_3way_cpu(
        self,
        n_samples: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform random 3-way split of data in CPU.
        Returns indices for train, val, and test sets.
        """
        np.random.seed(random_seed)

        indices = np.random.permutation(n_samples)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        return train_idx, val_idx, test_idx

    def _stratified_split_3way_cpu(
        self,
        y: np.ndarray,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform stratified 3-way split of data in CPU (maintains class distribution).
        Returns indices for train, val, and test sets.
        """
        np.random.seed(random_seed)

        unique_classes = np.unique(y)
        train_idx_list = []
        val_idx_list = []
        test_idx_list = []

        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            n_cls = len(cls_indices)

            cls_indices = cls_indices[np.random.permutation(n_cls)]

            train_end = int(n_cls * train_ratio)
            val_end = train_end + int(n_cls * val_ratio)

            train_idx_list.append(cls_indices[:train_end])
            val_idx_list.append(cls_indices[train_end:val_end])
            test_idx_list.append(cls_indices[val_end:])

        train_idx = np.concatenate(train_idx_list)
        val_idx = np.concatenate(val_idx_list)
        test_idx = np.concatenate(test_idx_list)

        train_idx = train_idx[np.random.permutation(len(train_idx))]
        val_idx = val_idx[np.random.permutation(len(val_idx))]
        test_idx = test_idx[np.random.permutation(len(test_idx))]

        return train_idx, val_idx, test_idx

    def get_train(self):
        """Get training data"""
        return self.X_train, self.y_train

    def get_val(self):
        """Get validation data"""
        return self.X_val, self.y_val

    def get_test(self):
        """Get test data"""
        return self.X_test, self.y_test
