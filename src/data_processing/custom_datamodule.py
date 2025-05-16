
from src.data_processing.base_datamodule import BaseDataModule
from typing import List, Tuple
import numpy as np

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

    # 🚧  Replace `_make_dataset` + the *_datasets methods with real logic
    def _make_dataset(self, rng) -> Tuple[np.ndarray, np.ndarray]:
        X = rng.normal(size=(self.n_samples, self.n_features))
        y = (X[:, 0] + rng.normal(scale=0.5, size=self.n_samples) > 0).astype(int)
        return X, y

    def _generate(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        rng = np.random.default_rng(self.random_seed)
        X_list, y_list = zip(*(self._make_dataset(rng) for _ in range(self.n_datasets)))
        return list(X_list), list(y_list)

    def train_datasets(self):
        return self._generate()

    def val_datasets(self):
        # Optional: supply explicit validation sets or just return empty lists.
        # We will sample from the training datasets in case None are provided!
        return [], []

    def test_datasets(self):
        return self._generate()