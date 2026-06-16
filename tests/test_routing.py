"""
test_routing.py
================
Unit tests for the YORO complexity scorer and router.
No external dependencies required (no API key, no DKL Excel).

Run:
    python -m pytest tests/test_routing.py -v
    # or
    python tests/test_routing.py
"""

import sys
import os
import unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yoro.yoro_hybrid_inference import (
    compute_complexity_score,
    route_question,
    InferencePath,
    YORO_PURE_THRESHOLD,
    YORO_HYBRID_THRESHOLD,
)


class TestComplexityScorer(unittest.TestCase):

    # --- Simple questions: should score low (Path A) ---

    def test_simple_top_n(self):
        q = "What are the top 10 customers by total sales in March 2018?"
        score, _ = compute_complexity_score(q)
        self.assertLess(score, YORO_PURE_THRESHOLD,
                        f"Expected simple query to score < {YORO_PURE_THRESHOLD}, got {score:.2f}")

    def test_simple_average(self):
        q = "What is the average review score for orders delivered in Q4 2017?"
        score, _ = compute_complexity_score(q)
        self.assertLess(score, YORO_PURE_THRESHOLD)

    def test_simple_count(self):
        q = "How many orders were placed in March 2018?"
        score, _ = compute_complexity_score(q)
        self.assertLess(score, YORO_PURE_THRESHOLD)

    # --- Geospatial: should push above YORO_PURE threshold ---

    def test_geospatial_above_pure(self):
        q = "Which customers are located more than 500km away from their seller?"
        score, _ = compute_complexity_score(q)
        self.assertGreaterEqual(score, YORO_PURE_THRESHOLD,
                                f"Geo query expected >= {YORO_PURE_THRESHOLD}, got {score:.2f}")

    # --- Partition window: should route to Path B ---

    def test_partition_window(self):
        q = "Which are the top 5 sellers by revenue in each state?"
        score, _ = compute_complexity_score(q)
        decision = route_question(q, yoro_available=True)
        self.assertEqual(decision.path, InferencePath.YORO_HYBRID,
                         f"Partitioned window expected Path B, score={score:.2f}")

    # --- Correlation: routes to B or C (above PURE threshold) ---

    def test_correlation_not_pure(self):
        q = "Is there a correlation between GMV and CSAT scores across product categories?"
        score, _ = compute_complexity_score(q)
        self.assertGreater(score, YORO_PURE_THRESHOLD,
                           f"Correlation query should be above pure threshold, got {score:.2f}")

    # --- Score bounds ---

    def test_score_bounds(self):
        """Score should always be in [0, 1]."""
        for q in [
            "a",
            "What are the top 10 customers by total sales in March 2018?",
            "Do the total payment values in the payments table match the sum of item prices plus freight AND do categories with higher GMV have better CSAT scores and what is the correlation between shipping distance and review score per state?",
        ]:
            score, _ = compute_complexity_score(q)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_complex_scores_higher_than_simple(self):
        """Complex questions should score higher than simple ones."""
        simple_q = "How many orders were placed in March 2018?"
        complex_q = "Is there a correlation between GMV and CSAT scores by category? Also check if total payment values match item prices plus freight, and identify customers more than 500km from their seller."
        simple_score, _ = compute_complexity_score(simple_q)
        complex_score, _ = compute_complexity_score(complex_q)
        self.assertGreater(complex_score, simple_score)


class TestRouter(unittest.TestCase):

    def test_simple_routes_to_yoro_pure(self):
        q = "What are the top 10 customers by total sales in March 2018?"
        decision = route_question(q, yoro_available=True)
        self.assertEqual(decision.path, InferencePath.YORO_PURE)

    def test_partition_routes_to_hybrid(self):
        q = "Which are the top 5 sellers by revenue in each state, excluding sellers with average rating < 3?"
        decision = route_question(q, yoro_available=True)
        # Should be B or C — not A
        self.assertNotEqual(decision.path, InferencePath.YORO_PURE,
                            "Partitioned window with filter should not be Path A")

    def test_unavailable_yoro_forces_graphrag(self):
        q = "What are the top 10 customers by total sales in March 2018?"
        decision = route_question(q, yoro_available=False)
        self.assertEqual(decision.path, InferencePath.GRAPHRAG_ONLY)

    def test_force_path_override(self):
        q = "What are the top 10 customers by total sales in March 2018?"
        decision = route_question(q, yoro_available=True,
                                  force_path=InferencePath.GRAPHRAG_ONLY)
        self.assertEqual(decision.path, InferencePath.GRAPHRAG_ONLY)
        self.assertEqual(decision.confidence, 1.0)

    def test_confidence_inversely_proportional_to_complexity(self):
        simple_q = "How many orders in March 2018?"
        complex_q = "Is there a correlation between GMV and CSAT across categories, and do customers more than 500km from sellers have lower review scores?"
        simple_dec = route_question(simple_q)
        complex_dec = route_question(complex_q)
        self.assertGreater(simple_dec.confidence, complex_dec.confidence)

    def test_decision_has_reasons(self):
        q = "What are the top 10 customers by total sales in March 2018?"
        decision = route_question(q)
        self.assertGreater(len(decision.reasons), 0)
        self.assertIn("complexity=", decision.reasons[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
