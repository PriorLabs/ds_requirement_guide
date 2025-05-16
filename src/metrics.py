
from enum import Enum
from typing import Callable, Dict, List, Tuple, Union
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
)
import numpy as np


class MetricType(str, Enum):
    """Metrics type

    Values:
        ACCURACY: Classification accuracy (proportion of correct predictions)
        ROC_AUC: Area under the ROC curve (for binary or multiclass problems)
        F1: F1 score (harmonic mean of precision and recall)
        RMSE: Root mean squared error (regression)
        MSE: Mean squared error (regression)
        MAE: Mean absolute error (regression)
    """

    ACCURACY = "accuracy"
    ROC_AUC = "roc_auc"
    LOG_LOSS = "log_loss"
    F1 = "f1"
    RMSE = "rmse"
    MSE = "mse"
    MAE = "mae"

def compute_accuracy(y_true, y_pred) -> float:
    return accuracy_score(y_true, y_pred)

def compute_roc_auc(y_true, y_pred_proba, multi_class="raise", average="macro") -> float:
    """Handles binary vs. multiclass AUROC transparently."""
    # Binary case — probability of positive class
    if y_pred_proba.ndim == 1 or y_pred_proba.shape[1] == 2:
        proba = y_pred_proba[:, 1] if y_pred_proba.ndim == 2 else y_pred_proba
        return roc_auc_score(y_true, proba)

    # Multiclass one‑vs‑rest
    return roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="macro")

def compute_log_loss(y_true, y_pred_proba) -> float:
    y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
    return log_loss(y_true, y_pred_proba, labels=np.unique(y_true))

def compute_f1(y_true, y_pred, average="macro") -> float:
    return f1_score(y_true, y_pred, average=average)

def compute_rmse(y_true, y_pred) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compute_mse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred)

def compute_mae(y_true, y_pred) -> float:
    return mean_absolute_error(y_true, y_pred)



def evaluate_metric(metric: MetricType, y_true, y_pred, y_score: Union[None, np.ndarray] = None) -> float:
    if metric == MetricType.ACCURACY:
        return compute_accuracy(y_true, y_pred)
    elif metric == MetricType.ROC_AUC:
        if y_score is None:
            raise ValueError("y_score must be provided for ROC AUC.")
        return compute_roc_auc(y_true, y_score)
    elif metric == MetricType.LOG_LOSS:
        if y_score is None:
            raise ValueError("y_score must be provided for log loss.")
        return compute_log_loss(y_true, y_score)
    elif metric == MetricType.F1:
        return compute_f1(y_true, y_pred)
    elif metric == MetricType.RMSE:
        return compute_rmse(y_true, y_pred)
    elif metric == MetricType.MSE:
        return compute_mse(y_true, y_pred)
    elif metric == MetricType.MAE:
        return compute_mae(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported metric: {metric}")