"""
tests/test_nutrition.py — Unit tests for nutrition lookup.
"""

from __future__ import annotations

import unittest
from app.nutrition import get_nutrition, get_calories, get_macro_summary


class TestNutrition(unittest.TestCase):

    def setUp(self) -> None:
        # Mock USDA query to prevent live network calls during tests
        import nutrition_db
        self.original_query_usda = getattr(nutrition_db, "_query_usda", None)
        if self.original_query_usda:
            nutrition_db._query_usda = lambda query: None

    def tearDown(self) -> None:
        import nutrition_db
        if self.original_query_usda:
            nutrition_db._query_usda = self.original_query_usda

    def test_get_nutrition_keys_exist(self) -> None:
        # Verify default key backfilling
        info = get_nutrition("nonexistent_food_item")
        self.assertIn("calories", info)
        self.assertIn("protein", info)
        self.assertIn("fat", info)
        self.assertIn("carbs", info)
        self.assertIn("fiber", info)
        self.assertIn("sugar", info)
        self.assertIn("sodium", info)
        self.assertIn("serving_g", info)

    def test_get_calories(self) -> None:
        # Check fallback to default or built-in values
        kcal = get_calories("pizza")
        self.assertEqual(kcal, 266)  # Built-in pizza value is 266

        kcal_unknown = get_calories("non_cached_gibberish_food_xyz")
        self.assertEqual(kcal_unknown, 200)  # Default fallback

    def test_get_macro_summary(self) -> None:
        summary = get_macro_summary("pizza")
        # serving_g for pizza is 200, so macros are doubled:
        # protein: 11.0 * 2 = 22g, fat: 10.0 * 2 = 20g, carbs: 33.0 * 2 = 66g
        self.assertIn("P 22.0g", summary)
        self.assertIn("F 20.0g", summary)
        self.assertIn("C 66.0g", summary)


if __name__ == "__main__":
    unittest.main()
