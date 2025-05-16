import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib.util
import pytest
from src.data_processing.base_datamodule import BaseDataModule

CUSTOM_MODULE_PATH = "src/data_processing/custom_datamodule.py"

MAX_TRAIN_SAMPLES = 5000
MAX_TEST_SAMPLES = 10000
MAX_FEATURES = 500


def load_custom_datamodule(path: str) -> BaseDataModule:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")

    spec = importlib.util.spec_from_file_location("custom_datamodule", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_datamodule"] = module
    spec.loader.exec_module(module)

    # Find subclass of BaseDataModule
    for attr in dir(module):
        obj = getattr(module, attr)
        if isinstance(obj, type) and issubclass(obj, BaseDataModule) and obj is not BaseDataModule:
            return obj()

    raise ValueError("No subclass of BaseDataModule found in custom_datamodule.py.")

def validate_datamodule(dm: BaseDataModule):
    X_train, y_train = dm.train_datasets()
    X_test, y_test = dm.test_datasets()

    assert isinstance(X_train, list), "train_datasets() should return a list of arrays"
    assert isinstance(y_train, list), "train_datasets() should return a list of arrays"
    assert len(X_train) == len(y_train), "Mismatch between X_train and y_train lengths"

    assert isinstance(X_test, list), "test_datasets() should return a list of arrays"
    assert isinstance(y_test, list), "test_datasets() should return a list of arrays"
    assert len(X_test) == len(y_test), "Mismatch between X_test and y_test lengths"

    # Shape checks with max limits
    for i, (x, y) in enumerate(zip(X_train, y_train)):
        assert x.shape[0] <= MAX_TRAIN_SAMPLES, f"X_train[{i}] has {x.shape[0]} samples, expected at most {MAX_TRAIN_SAMPLES}"
        assert x.shape[1] <= MAX_FEATURES, f"X_train[{i}] has {x.shape[1]} features, expected at most {MAX_FEATURES}"
        assert y.shape[0] == x.shape[0], f"y_train[{i}] does not match X_train[{i}] in samples"

    for i, (x, y) in enumerate(zip(X_test, y_test)):
        assert x.shape[0] <= MAX_TEST_SAMPLES, f"X_test[{i}] has {x.shape[0]} samples, expected at most {MAX_TEST_SAMPLES}"
        assert x.shape[1] <= MAX_FEATURES, f"X_test[{i}] has {x.shape[1]} features, expected at most {MAX_FEATURES}"
        assert y.shape[0] == x.shape[0], f"y_test[{i}] does not match X_test[{i}] in samples"

@pytest.mark.integration
def test_custom_datamodule_implements_interface():
    dm = load_custom_datamodule(CUSTOM_MODULE_PATH)
    assert isinstance(dm, BaseDataModule), "Loaded module is not a BaseDataModule"

@pytest.mark.integration
def test_custom_datamodule_validates_successfully():
    dm = load_custom_datamodule(CUSTOM_MODULE_PATH)
    validate_datamodule(dm)
