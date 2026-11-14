"""Checkerboard geometry, matched ASR protocol, and isolated run artifacts."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from src.utils.attack_report import prepare_output_directory, write_attack_report

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from src.data.backdoor import PoisonedDataset, create_asr_dataloader


@unittest.skipUnless(HAS_TORCH, "Torch is not installed")
class TestCheckerboard(unittest.TestCase):
    def dataset(self):
        return torch.utils.data.TensorDataset(torch.zeros(4, 3, 32, 32), torch.tensor([0, 1, 2, 0]))

    def test_exact_black_white_geometry_and_no_change_outside_patch(self):
        base = self.dataset()
        poisoned = PoisonedDataset(base, [1], 0, 4, "checkerboard")
        image, label = poisoned[1]
        patch = image[:, -4:, -4:] * poisoned.std + poisoned.mean
        expected = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.],
                                 [1., 0., 1., 0.], [0., 1., 0., 1.]])
        self.assertTrue(torch.allclose(patch, expected.expand(3, -1, -1), atol=1e-6))
        outside = torch.ones_like(image, dtype=torch.bool)
        outside[:, -4:, -4:] = False
        self.assertTrue(torch.equal(image[outside], base[1][0][outside]))
        self.assertTrue(torch.equal(base[1][0], torch.zeros(3, 32, 32)))
        self.assertEqual(label, 0)

    def test_unselected_image_and_label_are_unchanged(self):
        base = self.dataset()
        poisoned = PoisonedDataset(base, [1], 0, 4, "checkerboard")
        image, label = poisoned[2]
        self.assertTrue(torch.equal(image, base[2][0]))
        self.assertEqual(label, 2)

    def test_asr_filters_target_class_and_uses_identical_trigger(self):
        base = self.dataset()
        training = PoisonedDataset(base, [1], 0, 4, "checkerboard")
        loader = create_asr_dataloader(torch.utils.data.DataLoader(base), 0, 4, 2, "checkerboard")
        self.assertEqual(len(loader.dataset), 2)
        image, label = loader.dataset[0]
        self.assertTrue(torch.equal(image, training[1][0]))
        self.assertEqual(label, 0)
        self.assertTrue(all(int(y) == 0 for _, y in loader.dataset))

    def test_legacy_default_remains_white_three_by_three(self):
        base = self.dataset()
        poisoned = PoisonedDataset(base, [1], 0)
        image, _ = poisoned[1]
        self.assertEqual(poisoned.trigger_type, "white_patch")
        self.assertTrue(torch.allclose(image[:, -3:, -3:], poisoned.trigger_value))
        self.assertTrue(torch.equal(image[:, :-3, :], base[1][0][:, :-3, :]))

    def test_invalid_patterns_sizes_and_oversized_patch_are_rejected(self):
        with self.assertRaises(ValueError):
            PoisonedDataset(self.dataset(), [1], 0, 4, "unknown")
        with self.assertRaises(ValueError):
            PoisonedDataset(self.dataset(), [1], 0, 0, "checkerboard")
        with self.assertRaises(ValueError):
            PoisonedDataset(self.dataset(), [1], 0, 33, "checkerboard")[1]

    @unittest.skipUnless(importlib.util.find_spec("torchvision") and importlib.util.find_spec("yaml"),
                         "Torchvision/PyYAML are not installed")
    def test_client_zero_only_and_attack_start_round(self):
        from scripts.run_experiment_v2 import get_client_dataloader
        config = dict(seed=42, malicious_client_id=0, attack_start_round=30, batch_size=2,
                      target_class=0, trigger_size=4, trigger_type="checkerboard")
        base = self.dataset()
        before, active_before = get_client_dataloader(config, base, 0, 29, True, [1])
        during, active_during = get_client_dataloader(config, base, 0, 30, True, [1])
        benign, active_benign = get_client_dataloader(config, base, 1, 30, True, [1])
        self.assertFalse(active_before)
        self.assertIs(before.dataset, base)
        self.assertTrue(active_during)
        self.assertEqual(during.dataset.trigger_type, "checkerboard")
        self.assertEqual(during.dataset.poison_indices, {1})
        self.assertFalse(active_benign)
        self.assertIs(benign.dataset, base)


class TestAttackArtifacts(unittest.TestCase):
    def test_refuse_nonempty_output_without_altering_existing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "older_results"
            path.mkdir()
            artifact = path / "final_results.json"
            artifact.write_text('{"clean_accuracy": 75.01}')
            with self.assertRaises(ValueError):
                prepare_output_directory(str(path))
            self.assertEqual(json.loads(artifact.read_text()), {"clean_accuracy": 75.01})

    def test_new_or_empty_directory_is_accepted_but_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new_run"
            prepare_output_directory(str(output))
            prepare_output_directory(str(output))
            existing_file = Path(tmp) / "file.txt"
            existing_file.write_text("keep")
            with self.assertRaises(ValueError):
                prepare_output_directory(str(existing_file))

    def test_checkerboard_reruns_get_fresh_numbered_folders_and_preserve_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "iid_same_run_id"
            output.mkdir()
            (output / "config.json").write_text("original")
            first = Path(prepare_output_directory(output, unique_on_conflict=True))
            self.assertEqual(first.name, "iid_same_run_id_rerun_001")
            self.assertEqual(list(first.iterdir()), [])
            # Even an existing empty numbered sibling is not reused.
            second = Path(prepare_output_directory(output, unique_on_conflict=True))
            self.assertEqual(second.name, "iid_same_run_id_rerun_002")
            self.assertEqual((output / "config.json").read_text(), "original")
            self.assertEqual(prepare_output_directory(second, unique_on_conflict=True), str(second))
            existing_file = Path(tmp) / "not_a_folder"
            existing_file.write_text("keep")
            with self.assertRaises(ValueError):
                prepare_output_directory(existing_file, unique_on_conflict=True)

    def test_report_uses_final_endpoint_and_separates_peak_asr(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with (path / "metrics_per_round.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Round", "Clean_Acc", "Triggered_Target_Rate", "Untriggered_Airplane_Rate", "Attack_Active"])
                writer.writerow([1, 50, 99, 5, False])
                writer.writerow([30, 72, 90, 5, True])
                writer.writerow([50, 74, 80, 4, True])
            config = dict(experiment_name="checkerboard_test", trigger_type="checkerboard", trigger_size=4,
                          partition_type="iid", seed=42, target_class=0, malicious_client_id=0,
                          attack_start_round=30, num_rounds=50, num_clients=10, local_epochs=2,
                          batch_size=64, learning_rate=0.01, momentum=0.9)
            output = dict(config=config, mode="attack", clean_accuracy=74., triggered_rate=80.,
                          untriggered_airplane_rate=4., poisoned_samples=500, malicious_client_samples=5000)
            write_attack_report(str(path), output, "checkerboard.yaml", True)
            report = (path / "attack_report.md").read_text(encoding="utf-8")
            self.assertIn("74.00%", report)
            self.assertIn("80.00%", report)
            self.assertIn("90.00% at round 30", report)
            self.assertNotIn("99.00%", report)
            self.assertIn("W B W B\nB W B W", report)


if __name__ == "__main__":
    unittest.main()
