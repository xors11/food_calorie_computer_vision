"""
tests/test_estimation.py — Unit tests for serving size estimation.
"""

from __future__ import annotations

import unittest
from app.estimation import estimate_serving_multiplier, estimate_serving


class TestEstimation(unittest.TestCase):

    def test_serving_multiplier_buckets(self) -> None:
        # Test bbox ratio mapping to serving size multipliers
        self.assertEqual(estimate_serving_multiplier(0.02), 0.5)   # Tiny
        self.assertEqual(estimate_serving_multiplier(0.10), 1.0)   # Medium
        self.assertEqual(estimate_serving_multiplier(0.20), 1.5)   # Large
        self.assertEqual(estimate_serving_multiplier(0.50), 2.0)   # Very large

    def test_estimate_serving_scaling(self) -> None:
        # Mock nutrition info (per 100g)
        mock_nut = {
            "calories": 200,
            "protein": 10.0,
            "fat": 5.0,
            "carbs": 20.0,
            "fiber": 2.0,
            "sugar": 4.0,
            "sodium": 300,
            "serving_g": 150,
            "source": "test",
        }

        # Multiplier of 1.0 (medium detection)
        est = estimate_serving("pizza", "Pizza", 0.90, 0.10, mock_nut)
        self.assertEqual(est.serving_multiplier, 1.0)
        self.assertEqual(est.serving_g, 150.0)
        self.assertEqual(est.calories, 300.0)   # 200 * 1.5
        self.assertEqual(est.protein, 15.0)     # 10.0 * 1.5
        self.assertEqual(est.fat, 7.5)          # 5.0 * 1.5

        # Multiplier of 0.5 (tiny detection)
        est_tiny = estimate_serving("pizza", "Pizza", 0.90, 0.02, mock_nut)
        self.assertEqual(est_tiny.serving_multiplier, 0.5)
        self.assertEqual(est_tiny.serving_g, 75.0)
        self.assertEqual(est_tiny.calories, 150.0)
        self.assertEqual(est_tiny.protein, 7.5)


if __name__ == "__main__":
    unittest.main()
