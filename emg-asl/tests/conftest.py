"""
Shared pytest fixtures for the EMG-ASL test suite.

These fixtures are automatically available to every test module in the tests/
package without any explicit import.  Fixtures with scope='session' are
computed once per test run and reused, which keeps the full suite fast.

Fixture hierarchy:

  synthetic_session      (scope=session)  -- one DataFrame for label 'A'
  synthetic_windows      (scope=session)  -- (N, 40, 8) windows from that session
  tiny_lstm              (scope=function) -- small LSTM for fast forward-pass tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make sure the repository root is on sys.path so src.* imports work whether
# pytest is invoked from the repo root or from any subdirectory.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Session-scoped fixtures (computed once per pytest run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_session():
    """Generate one sEMG session for label 'A', cached for the whole test run.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns timestamp_ms, ch1..ch8, label.
        Hold rows have label='A', rest rows have label='rest'.
    """
    from src.data.synthetic import generate_session

    return generate_session(
        label="A",
        n_reps=4,
        fs=200,
        hold_duration_s=0.5,
        rest_duration_s=0.2,
        noise_level=0.1,
        rng=np.random.default_rng(42),
    )


@pytest.fixture(scope="session")
def synthetic_windows(synthetic_session):
    """Pre-compute sliding windows from the session DataFrame.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (windows, labels) where windows has shape (N, 40, 8) and labels
        is a 1-D string array of length N.  Only windows with a uniform
        non-rest label are included.
    """
    from src.data.loader import create_windows

    windows, labels = create_windows(
        session_df=synthetic_session,
        window_size=40,
        overlap=0.5,
        augment=False,
    )
    return windows, labels


# ---------------------------------------------------------------------------
# Function-scoped fixtures (a fresh instance per test)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_lstm():
    """Return a small ASLEMGClassifier for fast forward-pass / inference tests.

    Uses input_size=320, hidden_size=32, num_layers=1 so PyTorch keeps the
    tensor operations tiny.  Includes label_names so predict() returns
    human-readable strings.

    Returns
    -------
    ASLEMGClassifier
        In eval mode, random weights.
    """
    from src.models.lstm_classifier import ASLEMGClassifier
    from src.utils.constants import ASL_LABELS, NUM_CLASSES

    model = ASLEMGClassifier(
        input_size=320,
        hidden_size=32,
        num_layers=1,
        num_classes=NUM_CLASSES,
        dropout=0.0,  # 0 for single-layer LSTM (PyTorch requirement)
        label_names=list(ASL_LABELS),
    )
    model.eval()
    return model
