"""SVM baseline classifier using extracted EMG features."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.constants import N_CLASSES


def build_svm_pipeline(kernel: str = "rbf", C: float = 10.0, gamma: str = "scale") -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)),
    ])


def train_svm(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> Pipeline:
    """Train SVM on feature vectors. X_train: (N, feature_dim)."""
    pipe = build_svm_pipeline(**kwargs)
    pipe.fit(X_train, y_train)
    return pipe


def save_svm(pipe: Pipeline, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipe, f)


def load_svm(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)
