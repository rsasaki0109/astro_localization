#!/usr/bin/env python3
"""Unit tests for the raw-measurement observability core."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from predictive_localizability import (  # noqa: E402
    fuse_positions_directional,
    report_from_jacobian,
    skyline_grid_observability,
    skyline_measurement_jacobian,
    trn_observability,
)


class PredictiveLocalizabilityTest(unittest.TestCase):
    def test_report_marks_missing_direction_as_risky(self):
        report = report_from_jacobian(
            np.array([[2.0, 0.0], [0.0, 0.0]]),
            measurement_sigma=1.0,
            state_scale=(1.0, 1.0),
            state_labels=("x", "y"),
        )
        self.assertAlmostEqual(report.directional_confidence[0], 1.0)
        self.assertAlmostEqual(report.directional_confidence[1], 0.0)
        self.assertAlmostEqual(report.alignment_risk, 1.0)

    def test_skyline_jacobian_recovers_translation_and_yaw_columns(self):
        az = np.linspace(0.0, 2.0 * math.pi, 180, endpoint=False)

        def renderer(xy):
            return xy[0] * np.sin(az) + xy[1] * np.cos(2.0 * az)

        _, J = skyline_measurement_jacobian(
            renderer, (3.0, 4.0), 0.2,
            position_delta_m=0.1, yaw_delta_rad=1e-3,
        )
        self.assertEqual(J.shape, (180, 3))
        self.assertGreater(np.linalg.norm(J[:, 0]), 1.0)
        self.assertGreater(np.linalg.norm(J[:, 1]), 1.0)
        self.assertGreater(np.linalg.norm(J[:, 2]), 1.0)

    def test_grid_observability_reuses_profiles(self):
        az = np.linspace(0.0, 2.0 * math.pi, 60, endpoint=False)
        profiles = []
        for y in range(3):
            for x in range(3):
                profiles.append(x * np.sin(az) + y * np.cos(2.0 * az))
        reliability, reports = skyline_grid_observability(
            np.asarray(profiles),
            grid_shape=(3, 3),
            grid_step_m=2.0,
            horizon_sigma_rad=0.01,
            yaw_scale_rad=0.1,
        )
        self.assertEqual(reliability.shape, (3, 3))
        self.assertEqual(len(reports), 9)
        self.assertTrue(np.all(np.isfinite(reliability)))

    def test_flat_trn_response_is_unobservable(self):
        response = np.ones((9, 9), dtype=np.float64)
        report = trn_observability(response, (4, 4), px_to_m=2.0)
        np.testing.assert_allclose(report.eigenvalues, 0.0)
        self.assertAlmostEqual(report.alignment_risk, 1.0)

    def test_trn_reports_one_direction_cliff(self):
        response = np.zeros((9, 9), dtype=np.float64)
        response[4, 4] = 1.0
        response[4, 3] = 0.0
        response[4, 5] = 0.0
        response[3, 4] = 1.0
        response[5, 4] = 1.0
        report = trn_observability(response, (4, 4), px_to_m=1.0)
        self.assertGreater(report.directional_confidence[0], 0.9)
        self.assertAlmostEqual(report.directional_confidence[1], 0.0)
        self.assertAlmostEqual(report.alignment_risk, 1.0)

    def test_directional_factor_only_pulls_observed_axis(self):
        estimate, covariance = fuse_positions_directional(
            2,
            np.array([[1.0, 1.0]]),
            (0.0, 0.0),
            sigma_prior=10.0,
            sigma_vo=1.0,
            fixes=[(
                1,
                np.array([5.0, 9.0]),
                np.array([[100.0, 0.0], [0.0, 0.0]]),
            )],
        )
        self.assertAlmostEqual(estimate[1, 0], 5.0, places=1)
        self.assertAlmostEqual(estimate[1, 1], 1.0, places=6)
        self.assertLess(covariance[1, 0, 0], covariance[1, 1, 1])


if __name__ == "__main__":
    unittest.main()
