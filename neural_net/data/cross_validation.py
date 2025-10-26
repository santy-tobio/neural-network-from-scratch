from collections.abc import Generator
import cupy as cp


class KFoldSplitter:
    """
    K-Fold cross-validation splitter.
    Works with Dataset objects to generate train/val splits.
    """

    def __init__(self, n_splits: int, random_seed: int = 42, stratified: bool = True):
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.stratified = stratified

    def split(
        self, X: cp.ndarray, y: cp.ndarray
    ) -> Generator[tuple[cp.ndarray, cp.ndarray], None, None]:
        """
        Generate K-Fold splits.
        """
        n_samples = X.shape[0]

        if self.stratified:
            fold_indices = self._stratified_kfold(y, n_samples)
        else:
            fold_indices = self._random_kfold(n_samples)

        # Generate train/val splits for each fold
        for fold in range(self.n_splits):
            val_idx = fold_indices[fold]
            train_idx = cp.concatenate(
                [fold_indices[i] for i in range(self.n_splits) if i != fold]
            )
            yield train_idx, val_idx

    def _get_all_folds(
        self, X: cp.ndarray, y: cp.ndarray
    ) -> list[tuple[cp.ndarray, cp.ndarray]]:
        """
        (Utility) Get all K-Fold splits as a list.
        """
        return list(self.split(X, y))

    def _random_kfold(self, n_samples: int) -> list[cp.ndarray]:
        """
        Random K-Fold split implementation.

        Args:
            n_samples: Total number of samples

        Returns:
            List of indices arrays, one for each fold
        """
        cp.random.seed(self.random_seed)
        indices = cp.random.permutation(n_samples)

        # Split indices into n_splits folds
        fold_sizes = cp.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1  # Distribute remainder

        fold_indices = []
        current = 0
        for fold_size in fold_sizes:
            fold_indices.append(indices[current : current + fold_size])
            current += fold_size

        return fold_indices

    def _stratified_kfold(self, y: cp.ndarray, n_samples: int) -> list[cp.ndarray]:
        """
        Stratified K-Fold split implementation.
        Stratified K-Fold split:
        Maintains class distribution in each fold.
        Returns list of indices for each fold.
        """
        cp.random.seed(self.random_seed)

        # Get unique classes
        unique_classes = cp.unique(y)

        # Initialize fold lists
        fold_indices = [[] for _ in range(self.n_splits)]

        # For each class, distribute samples across folds
        for cls in unique_classes:
            cls_indices = cp.where(y == cls)[0]
            n_cls = len(cls_indices)

            # Shuffle class indices
            cls_indices = cls_indices[cp.random.permutation(n_cls)]

            # Calculate fold sizes for this class
            fold_sizes = cp.full(self.n_splits, n_cls // self.n_splits, dtype=int)
            fold_sizes[: n_cls % self.n_splits] += 1  # Distribute remainder

            # Distribute class samples to folds
            current = 0
            for fold, fold_size in enumerate(fold_sizes):
                fold_indices[fold].append(cls_indices[current : current + fold_size])
                current += fold_size

        # Concatenate indices for each fold and shuffle
        result_folds = []
        for fold_idx_list in fold_indices:
            fold_array = cp.concatenate(fold_idx_list)
            # Shuffle indices within each fold
            fold_array = fold_array[cp.random.permutation(len(fold_array))]
            result_folds.append(fold_array)

        return result_folds
