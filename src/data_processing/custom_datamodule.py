
from src.data_processing.base_datamodule import BaseDataModule
from typing import List, Tuple
import numpy as np

# 🚧  Replace `make_dataset_helper` + the *_datasets methods with real logic
def make_dataset_helper(rng, n_samples: int, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    X = rng.normal(size=(n_samples, n_features))
    y = (X[:, 0] + rng.normal(scale=0.5, size=n_samples) > 0).astype(int)
    return X, y


class ExampleDataModule(BaseDataModule):
    """Synthetic placeholder so the skeleton runs out‑of‑the‑box."""

    def __init__(
        self,
        n_datasets: int = 3,
        n_samples: int = 5000,
        n_features: int = 100,
        random_seed: int = 42,
    ):
        super().__init__(random_seed)
        self.n_datasets = n_datasets
        self.n_samples = n_samples
        self.n_features = n_features


    def train_datasets(self):
        rng = np.random.default_rng(self.random_seed)
        datasets = [make_dataset_helper(rng, self.n_samples, self.n_features) for _ in range(self.n_datasets)]
        X_list, y_list = zip(*datasets)
        return list(X_list), list(y_list)

    def val_datasets(self):
        # Optional: supply explicit validation sets or just return empty lists.
        # We will sample from the training datasets in case None are provided!
        return [], []

    def test_datasets(self):
        rng = np.random.default_rng(self.random_seed + 1)  # different seed for test
        datasets = [make_dataset_helper(rng, self.n_samples, self.n_features) for _ in range(self.n_datasets)]
        X_list, y_list = zip(*datasets)
        return list(X_list), list(y_list)