import os
import cupy as cp


class Dataset:
    """
    Handles data loading and splitting into train/val/test.
    This ensures ALL experiments use the same data splits,
    making comparisons fair.
    """

    def __init__(
        self,
        X_path: str,
        y_path: str,
        dev_ratio: float = 0.7,
        normalize: bool = True,
        stratify: bool = False,
        random_seed: int = 42,
    ):
        """
        Load and split dataset into dev/train.
        Dataset can be split randomly or in a stratified manner.
        Features can be normalized to [0, 1].
        """
        assert dev_ratio > 0 and dev_ratio <= 1, "Dev ratio must be in (0, 1]"
        assert os.path.exists(X_path) and os.path.exists(
            y_path
        ), "Data files must exist"

        self.X = cp.load(X_path)  # Shape: (n_samples, height, width)
        self.y = cp.load(y_path)  # Shape: (n_samples,)

        # Reshape X to (n_samples, height*width)
        self.X = self.X.reshape(self.X.shape[0], -1)

        if normalize:
            self.X = self.X.astype(cp.float32) / self.X.max()

        # Get indices from split method
        print(
            f"Splitting dataset: {self.X.shape[0]} samples - dev: {dev_ratio:.4f}, test: {1 - dev_ratio:.4f}"
        )
        if stratify:
            devel_idx, test_idx = self._stratified_split(dev_ratio, random_seed)
        else:
            devel_idx, test_idx = self._random_split(dev_ratio, random_seed)

        # Assign devel/test splits using indices
        self.X_devel = self.X[devel_idx]
        self.y_devel = self.y[devel_idx]

        self.X_test = self.X[test_idx]
        self.y_test = self.y[test_idx]

        print("Dataset loaded:")
        print(f"  Train: {len(self.X_devel)} samples")
        print(f"  Test: {len(self.X_test)} samples")

    def _random_split(
        self, devel_ratio: int, random_seed: int
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """
        Perform random split of data.
        Returns indices for devel and test sets.
        """
        cp.random.seed(random_seed)
        n_samples = self.X.shape[0]

        indices = cp.random.permutation(n_samples)
        devel_end = int(n_samples * devel_ratio)
        devel_idx = indices[:devel_end]
        test_idx = indices[devel_end:]

        return devel_idx, test_idx

    def _stratified_split(
        self, devel_ratio: int, random_seed: int
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """
        Perform stratified split of data (maintains class distribution).
        Returns indices for devel and test sets.
        """
        cp.random.seed(random_seed)

        # Get unique classes and their counts
        unique_classes = cp.unique(self.y)
        devel_idx_list = []
        test_idx_list = []

        # For each class, split proportionally
        for cls in unique_classes:
            cls_indices = cp.where(self.y == cls)[0]
            n_cls = len(cls_indices)

            # Shuffle class indices
            cls_indices = cls_indices[cp.random.permutation(n_cls)]

            # Calculate split point
            devel_end = int(n_cls * devel_ratio)

            # Split for this class
            devel_idx_list.append(cls_indices[:devel_end])
            test_idx_list.append(cls_indices[devel_end:])

        # Concatenate all class indices and shuffle
        devel_idx = cp.concatenate(devel_idx_list)
        test_idx = cp.concatenate(test_idx_list)

        # Shuffle each split
        devel_idx = devel_idx[cp.random.permutation(len(devel_idx))]
        test_idx = test_idx[cp.random.permutation(len(test_idx))]

        return devel_idx, test_idx

    def get_devel(self):
        """Get development/training data"""
        return self.X_devel, self.y_devel

    def get_test(self):
        """Get test data"""
        return self.X_test, self.y_test
