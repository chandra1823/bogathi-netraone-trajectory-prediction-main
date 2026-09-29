"""Tests for the from-scratch segment intersection geometry."""
from __future__ import annotations

import numpy as np
import pytest

from predict_breach import (fence_segments, first_path_crossing, orientation, point_in_polygon,
                            segment_intersection, segments_intersect)


def test_orientation() -> None:
    """Counter-clockwise, clockwise and collinear triplets."""
    assert orientation((0, 0), (1, 0), (1, 1)) == 1
    assert orientation((0, 0), (1, 0), (1, -1)) == -1
    assert orientation((0, 0), (1, 1), (2, 2)) == 0


def test_proper_crossing() -> None:
    """An X-shaped crossing is detected at the right point and fraction."""
    assert segments_intersect((0, 0), (2, 2), (0, 2), (2, 0))
    alpha, pt = segment_intersection((0, 0), (2, 2), (0, 2), (2, 0))
    assert alpha == pytest.approx(0.5)
    assert pt == pytest.approx((1.0, 1.0))


def test_parallel_no_intersection() -> None:
    """Parallel, non-collinear segments never intersect."""
    assert not segments_intersect((0, 0), (10, 0), (0, 1), (10, 1))
    assert segment_intersection((0, 0), (10, 0), (0, 1), (10, 1)) is None


def test_collinear_overlap() -> None:
    """Collinear overlapping segments intersect at the first overlap point."""
    assert segments_intersect((0, 0), (10, 0), (5, 0), (20, 0))
    alpha, pt = segment_intersection((0, 0), (10, 0), (5, 0), (20, 0))
    assert alpha == pytest.approx(0.5)
    assert pt == pytest.approx((5.0, 0.0))


def test_collinear_disjoint() -> None:
    """Collinear but disjoint segments do not intersect."""
    assert not segments_intersect((0, 0), (1, 0), (2, 0), (3, 0))


def test_touching_endpoint() -> None:
    """A path segment ending exactly on the fence counts as contact (alpha = 1)."""
    assert segments_intersect((5, 0), (5, 10), (0, 10), (10, 10))
    alpha, pt = segment_intersection((5, 0), (5, 10), (0, 10), (10, 10))
    assert alpha == pytest.approx(1.0)
    assert pt == pytest.approx((5.0, 10.0))
    # touching the fence post (fence endpoint)
    assert segments_intersect((0, 0), (10, 10), (10, 10), (20, 10))


def test_miss() -> None:
    """Segments that would cross only if extended do not intersect."""
    assert not segments_intersect((0, 0), (1, 1), (3, 0), (0, 3.5))
    assert not segments_intersect((0, 0), (1, 0), (2, -1), (2, 1))


def test_polyline_and_polygon_fence() -> None:
    """fence_segments builds open polylines and closed polygons."""
    pts = [(0, 0), (10, 0), (10, 10)]
    assert len(fence_segments(pts)) == 2
    assert len(fence_segments(pts, closed=True)) == 3
    path = np.array([[5.0, -5.0], [5.0, -1.0], [5.0, 3.0], [15.0, 5.0]])
    k, alpha, pt = first_path_crossing(path, fence_segments(pts))
    assert k == 1 and alpha == pytest.approx(0.25) and pt == pytest.approx((5.0, 0.0))


def test_point_in_polygon() -> None:
    """Ray-casting inside / outside test."""
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), sq)
    assert not point_in_polygon((15, 5), sq)
    assert not point_in_polygon((5, -1), sq)
