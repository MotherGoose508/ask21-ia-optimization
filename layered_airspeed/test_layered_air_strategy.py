from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import layered_air_strategy as strategy


ROOT = Path(__file__).resolve().parent


class LayeredStrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = strategy.load_model(ROOT / "model_parameters.csv")
        cls.layers = strategy.load_layers(ROOT / "layers.csv")
        cls.branches = strategy.make_branches(cls.model)

    def test_transition_is_continuous(self) -> None:
        lower, higher = self.branches
        speed = self.model.transition_speed_mps
        self.assertAlmostEqual(lower.sink_rate(speed), higher.sink_rate(speed), places=12)
        self.assertAlmostEqual(lower.derivative(speed), higher.derivative(speed), places=12)

    def test_still_air_optimum_matches_ia(self) -> None:
        layer = strategy.Layer("still air", 800.0, 0.0, 0.0)
        optimum, _ = strategy.optimize_layer(self.branches, layer)
        self.assertEqual(optimum.branch_code, "s1")
        self.assertAlmostEqual(optimum.speed_mps, 25.40399, places=5)
        self.assertAlmostEqual(optimum.range_ratio, 33.50285, places=5)

    def test_layered_strategy_cannot_lose_to_best_constant_speed(self) -> None:
        _, _, comparison, _ = strategy.calculate(self.model, self.layers)
        self.assertGreaterEqual(
            comparison["layer_specific_total_distance_m"] + 1e-8,
            comparison["optimized_constant_total_distance_m"],
        )

    def test_strong_layer_selects_higher_branch(self) -> None:
        optimum, _ = strategy.optimize_layer(self.branches, self.layers[-1])
        self.assertEqual(optimum.branch_code, "s2")
        self.assertAlmostEqual(optimum.speed_mps, 37.566625, places=5)

    def test_run_writes_requested_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            comparison = strategy.run(
                ROOT / "model_parameters.csv", ROOT / "layers.csv", output
            )
            expected = {
                "layer_results.csv",
                "layer_branch_candidates.csv",
                "strategy_comparison.csv",
                "verification.txt",
                "summary.txt",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            with (output / "layer_results.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(self.layers))
            self.assertTrue(all(row["selected_branch"] for row in rows))
            distance_sum = sum(float(row["layer_specific_distance_m"]) for row in rows)
            self.assertAlmostEqual(
                distance_sum, comparison["layer_specific_total_distance_m"], places=8
            )


if __name__ == "__main__":
    unittest.main()
