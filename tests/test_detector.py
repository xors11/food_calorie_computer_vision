"""
tests/test_detector.py — Unit tests for detector logic and NMS duplicate merging.
"""

from __future__ import annotations

import unittest
from app.detector import FoodDetector, Detection


class TestDetector(unittest.TestCase):

    def test_merge_overlapping_boxes(self) -> None:
        # Create two overlapping boxes of the same class (pizza)
        # Bbox a: (0, 0, 100, 100)
        det_a = Detection(
            label="pizza", display="Pizza", confidence=0.85,
            x1=0, y1=0, x2=100, y2=100, bbox_area_ratio=0.01
        )
        # Bbox b: (10, 10, 90, 90) - fully nested within a
        det_b = Detection(
            label="pizza", display="Pizza", confidence=0.90,
            x1=10, y1=10, x2=90, y2=90, bbox_area_ratio=0.01
        )

        merged = FoodDetector._merge_overlapping([det_a, det_b])
        # Should keep the higher confidence one (det_b, 90%)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].confidence, 0.90)

    def test_no_merge_different_classes(self) -> None:
        # Overlapping boxes of DIFFERENT classes should not be merged
        det_a = Detection(
            label="pizza", display="Pizza", confidence=0.85,
            x1=0, y1=0, x2=100, y2=100, bbox_area_ratio=0.01
        )
        det_b = Detection(
            label="apple", display="Apple", confidence=0.90,
            x1=10, y1=10, x2=90, y2=90, bbox_area_ratio=0.01
        )

        merged = FoodDetector._merge_overlapping([det_a, det_b])
        self.assertEqual(len(merged), 2)

    def test_no_merge_separate_regions(self) -> None:
        # Non-overlapping boxes of the SAME class should not be merged (e.g. two separate pizzas)
        det_a = Detection(
            label="pizza", display="Pizza", confidence=0.85,
            x1=0, y1=0, x2=100, y2=100, bbox_area_ratio=0.01
        )
        det_b = Detection(
            label="pizza", display="Pizza", confidence=0.90,
            x1=200, y1=200, x2=300, y2=300, bbox_area_ratio=0.01
        )

        merged = FoodDetector._merge_overlapping([det_a, det_b])
        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()
