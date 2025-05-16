


import numpy as np
import xgboost as xgb

from tqdm import tqdm
from typing import (
    Callable,
    Tuple,
    List,
    Dict,
)

from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.metrics import evaluate_metric, MetricType






def train_and_predict_single_dataset(
    model: xgb.XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fits *model* on the given split and returns encoded truth/ preds/ probas."""
    try:
        le = LabelEncoder().fit(y_train)
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)

        model.fit(X_train, y_train_enc)

        y_pred_enc = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        return y_test_enc, y_pred_enc, y_pred_proba
    except Exception as exc:  # pragma: no cover (debug aid)
        warnings.warn(f"Error during single‑dataset fit/predict: {exc}")
        return np.array([]), np.array([]), np.array([])


def benchmark_datasets(
    model: xgb.XGBClassifier,
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    *,
    split_fn: Callable = lambda X, y: train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    ),
) -> Dict[str, float]:
    """Runs *model* on each dataset & returns mean metric values."""

    if is_classifier(model):
        metrics_to_calculate = [MetricType.ACCURACY, MetricType.ROC_AUC, MetricType.F1, MetricType.LOG_LOSS]
    elif is_regressor(model):
        metrics_to_calculate = [MetricType.RMSE, MetricType.MSE, MetricType.MAE]
    else:
        raise ValueError("Model must be a classifier or regressor.")

    results: Dict[str, List[float]] = {m: [] for m in metrics_to_calculate}

    for X, y in tqdm(zip(X_list, y_list), total=len(X_list), desc="Datasets"):
        X_train, X_test, y_train, y_test = split_fn(X, y)
        y_true, y_pred, y_pred_proba = train_and_predict_single_dataset(
            model, X_train, y_train, X_test, y_test
        )

        for m in metrics_to_calculate:
            try:
                results[m].append(evaluate_metric(m, y_true, y_pred, y_pred_proba))
            except Exception as exc:
                warnings.warn(f"Failed to compute {m}: {exc}")
                results[m].append(np.nan)

    return {m: float(np.nanmean(v)) for m, v in results.items()}