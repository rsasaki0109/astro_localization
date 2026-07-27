#!/usr/bin/env python3
"""Integration test from the real template matcher to directional TRN info."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from four_factor_fusion_demo import TrnLocalizer  # noqa: E402


class TrnPredictiveObservabilityTest(unittest.TestCase):
    def test_matcher_returns_xy_observability_without_breaking_legacy_fix(self):
        yy, xx = np.mgrid[-32:32, -32:32]
        appearance = (
            np.exp(-((xx - 8) ** 2 + (yy + 5) ** 2) / 40.0)
            + 0.6 * np.exp(-((xx + 14) ** 2 + (yy - 11) ** 2) / 18.0)
            + 0.1 * np.sin(xx / 3.0)
        ).astype(np.float32)
        localizer = TrnLocalizer(appearance, 1.0, patch_m=14.0)

        legacy = localizer.fix((32.0, 32.0), np.random.default_rng(5), 0.0)
        directional = localizer.fix_with_observability(
            (32.0, 32.0), np.random.default_rng(5), 0.0
        )

        self.assertEqual(len(legacy), 3)
        self.assertEqual(len(directional), 4)
        np.testing.assert_allclose(legacy[0], directional[0])
        self.assertEqual(directional[3].state_labels, ("x", "y"))
        self.assertEqual(directional[3].information_matrix.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(directional[3].eigenvalues)))


if __name__ == "__main__":
    unittest.main()
