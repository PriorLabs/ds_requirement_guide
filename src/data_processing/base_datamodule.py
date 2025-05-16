from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np

class BaseDataModule(ABC):
    """Abstract interface that supplies multiple datasets as (X_list, y_list)."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    @abstractmethod
    def train_datasets(self) -> Tuple[List[np.ndarray], List[np.ndarray]]: 
        pass

    @abstractmethod
    def val_datasets(self) -> Tuple[List[np.ndarray], List[np.ndarray]]: 
        pass

    @abstractmethod
    def test_datasets(self) -> Tuple[List[np.ndarray], List[np.ndarray]]: 
        pass