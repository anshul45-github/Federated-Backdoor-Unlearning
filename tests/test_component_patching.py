"""Selection logic works without Torch; tensor invariants require Torch."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.unlearning.component_patching import (CHANNEL_KEYS, channel_rankings, choose_prefix,
    choose_stage1_setting, effects, stratified_split)

spec = importlib.util.spec_from_file_location("component_runner", ROOT / "scripts/run_component_patching.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class SelectionTests(unittest.TestCase):
    def test_disjoint_reproducible_stratified_split(self):
        labels = [c for c in range(10) for _ in range(1000)]
        selection, report = stratified_split(labels)
        self.assertEqual((selection, report), stratified_split(labels))
        self.assertFalse(set(selection) & set(report))
        self.assertEqual(sorted(selection + report), list(range(10000)))
        for indices in (selection, report):
            self.assertEqual([sum(labels[i] == c for i in indices) for c in range(10)], [500] * 10)
        self.assertNotEqual(selection, stratified_split(labels, seed=7)[0])

    def test_signed_deltas_and_zero_loss_score(self):
        row = effects(74, 10, {"ca": 75, "asr": 90})
        self.assertEqual((row["delta_ca_pp"], row["delta_asr_pp"], row["score"]), (-1, -80, 80))
        self.assertEqual(effects(76, 80, {"ca": 75, "asr": 90})["score"], 10 / 1e-6)
        with self.assertRaises(ValueError):
            effects(float("nan"), 0, {"ca": 75, "asr": 90})

    def test_rankings_and_random_controls(self):
        rows = [{"channel": c, "score": c, "asr_drop_pp": c, "ca_lost_pp": 1,
                 "weight_drift": 64 - c} for c in range(64)]
        ranks = channel_rankings(rows)
        self.assertEqual(len(ranks), 12)
        self.assertEqual(ranks["single_channel_score"], list(reversed(range(64))))
        self.assertEqual(ranks["weight_drift"], list(range(64)))
        self.assertEqual(ranks, channel_rankings(rows))
        for order in ranks.values():
            self.assertEqual(sorted(order), list(range(64)))
        self.assertEqual(len({tuple(v) for k, v in ranks.items() if k.startswith("random_")}), 10)

    def test_stage1_uses_score_and_deterministic_ties(self):
        rows = [dict(donor_round=r, alpha=.5, **effects(ca, asr, {"ca": 75, "asr": 90}))
                for r, ca, asr in [(10, 70, 10), (20, 74, 60), (29, 74, 60)]]
        self.assertEqual(choose_stage1_setting(rows)["donor_round"], 29)

    def test_budget_rejects_low_asr_with_excess_ca_loss(self):
        def row(ca, asr, k, ranking="single_channel_score"):
            return dict(ca=ca, asr=asr, prefix_k=k, ranking=ranking, donor_round=10, alpha=.5)
        rows = [row(75, 90, 0), row(73.98, 0, 5), row(74, 20, 4), row(74.5, 20, 3), row(74.5, 20, 2)]
        self.assertEqual(choose_prefix(rows, {"ca": 75}, 1)["prefix_k"], 2)
        self.assertEqual(choose_prefix(rows[:2], {"ca": 75}, 1)["prefix_k"], 0)

    def test_cli_requires_explicit_setting_for_stage2_only(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parse_args(["--source_run", "unused", "--stage", "2"])

    def test_search_outputs_complete_curves_and_selects_feasible_prefix(self):
        args = runner.parse_args(["--source_run", "unused", "--donor_rounds", "10", "--alphas", ".5"])
        baseline = {"ca": 75, "asr": 90}
        class Model:
            def load_state_dict(self, value):
                self.channels = value
        def measure(model):
            k = len(model.channels)
            return {"ca": 75 - k * .1, "asr": 90 - k}
        model = Model()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(runner, "patch_channels", side_effect=lambda f, d, c, a: list(c)), \
             patch.object(runner, "weight_drift", return_value=list(range(64))):
            output = Path(directory)
            selected, curves = runner.search(None, model, {}, {10: {}}, baseline, measure, args, output)
            self.assertEqual(selected["prefix_k"], 10)
            self.assertEqual(len(curves), 12 * 65)
            self.assertEqual(len((output / "single_channel_scores.csv").read_text().splitlines()), 65)
            self.assertEqual(len((output / "channel_curves.csv").read_text().splitlines()), 781)
            self.assertEqual(len(json.loads((output / "ranking_orders.json").read_text())["G10_alpha0.5"]), 12)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is not installed")
class TensorTests(unittest.TestCase):
    def setUp(self):
        import torch
        from src.models.small_cnn import SmallCNN
        self.torch = torch
        self.final = {k: v.clone() for k, v in SmallCNN().state_dict().items()}
        self.donor = {k: v.clone() for k, v in self.final.items()}
        for key in CHANNEL_KEYS:
            self.donor[key] += 2
        self.donor["features.5.num_batches_tracked"] += 100

    def test_interpolation_and_untouched_slices_and_inputs(self):
        from src.unlearning.component_patching import patch_channels, assert_only_selected_changed
        original = {k: v.clone() for k, v in self.final.items()}
        result = patch_channels(self.final, self.donor, [0, 17, 63], .25)
        assert_only_selected_changed(self.final, result, [0, 17, 63])
        for key in CHANNEL_KEYS:
            self.assertTrue(self.torch.allclose(result[key][[0, 17, 63]], self.final[key][[0, 17, 63]] + .5))
        for key in original:
            self.assertTrue(self.torch.equal(original[key], self.final[key]))
        self.assertTrue(self.torch.equal(result["features.5.num_batches_tracked"], original["features.5.num_batches_tracked"]))

    def test_full_replacement_and_identity(self):
        from src.unlearning.component_patching import patch_channels
        full = patch_channels(self.final, self.donor, list(range(64)), 1)
        for key in CHANNEL_KEYS:
            self.assertTrue(self.torch.equal(full[key], self.donor[key]))
        for channels, alpha in [([], 1), ([4], 0)]:
            result = patch_channels(self.final, self.donor, channels, alpha)
            self.assertTrue(all(self.torch.equal(result[k], v) for k, v in self.final.items()))

    def test_invalid_channels_and_drift(self):
        from src.unlearning.component_patching import patch_channels, weight_drift
        for channels in [[64], [-1], [1, 1], [True]]:
            with self.assertRaises(ValueError):
                patch_channels(self.final, self.donor, channels, .5)
        self.assertAlmostEqual(weight_drift(self.final, self.donor)[0], (4 * (32 * 3 * 3 + 1)) ** .5, places=4)

    def test_evaluation_respects_explicit_indices(self):
        torch = self.torch
        class Model(torch.nn.Module):
            def forward(self, images):
                if not torch.isfinite(images).all():
                    raise AssertionError("Evaluator accessed excluded images")
                logits = torch.zeros(len(images), 10)
                logits[:, 0] = 1
                return logits
        images = torch.zeros(4, 3, 32, 32)
        images[2:] = float("nan")  # Excluded images must never be accessed.
        result = runner.evaluate_indices(torch, Model(), images, torch.tensor([0, 1, 2, 3]), [0, 1],
            {"target_class": 0, "trigger_size": 4, "trigger_type": "checkerboard"}, "cpu", 1)
        self.assertEqual(result, {"ca": 50, "asr": 100, "clean_count": 2, "triggered_count": 1})


if __name__ == "__main__":
    unittest.main()
