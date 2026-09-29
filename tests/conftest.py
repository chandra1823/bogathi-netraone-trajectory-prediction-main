"""Shared pytest fixtures."""
from __future__ import annotations

import pandas as pd
import pytest

from generate_synthetic_tracks import generate_dataset


@pytest.fixture(scope="session")
def small_tracks() -> pd.DataFrame:
    """A small in-memory synthetic dataset (independent of data/tracks.csv)."""
    return generate_dataset(num_tracks=120, seed=7)
