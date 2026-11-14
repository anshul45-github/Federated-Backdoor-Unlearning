"""Checkerboard source routing/provenance and unlearning invariants."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

from scripts.run_checkerboard_unlearning import (aggregate_reconstruction, evaluate, main, parse_args,
                                                sha256, subtract, validate_inputs, write_results)


class Scalar:
    """A scalar for dependency-free testing of subtraction and integer bypass."""
    def __init__(self, value, floating=True):
        self.value, self.floating = value, floating
    def is_floating_point(self):
        return self.floating
    def __rmul__(self, value):
        return Scalar(value * self.value)
    def __sub__(self, other):
        return Scalar(self.value - other.value)


class TestVariantInputs(unittest.TestCase):
    def make_source(self, root, part="iid"):
        source = root / part
        source.mkdir()
        config = dict(trigger_type="checkerboard", trigger_size=4, partition_type=part, num_clients=10,
                      clients_per_round=10, malicious_client_id=0, num_rounds=2, seed=42,
                      dirichlet_alpha=0.5, local_epochs=2, learning_rate=0.01, momentum=0.9, batch_size=64,
                      target_class=0)
        (source / "config.json").write_text(json.dumps(config))
        (source / "final_results.json").write_text(json.dumps(dict(config=config, mode="attack", clean_accuracy=75.26, triggered_rate=92.18)))
        (source / "initial_model.pt").write_bytes(b"fixture initial")
        (source / "final_model.pt").write_bytes(b"fixture final")
        (source / "client_class_histograms.json").write_text(json.dumps({f"client_{c}": {"num_samples": 5000} for c in range(10)}))
        (source / "client_partitions.pt").write_bytes(b"fixture partitions")
        for r in [1, 2]:
            folder = source / "trajectory" / f"round_{r:03d}"
            folder.mkdir(parents=True)
            for c in range(10):
                (folder / f"delta_client_{c:02d}.pt").write_bytes(b"fixture update")
        return source

    def args(self, source, method, *extra):
        return parse_args(["--source_run", str(source), "--method", method, *extra])

    def test_iid_and_noniid_sources_validate_without_importing_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            for part in ["iid", "noniid"]:
                source = self.make_source(Path(tmp), part)
                result = validate_inputs(self.args(source, "subtraction"))
                self.assertEqual(result[1]["partition_type"], part)
                self.assertEqual(result[4], sha256(source / "final_model.pt"))
                self.assertEqual(sum(result[3]), 50000)


    def test_extreme_skew_accepts_empty_retained_client_but_rejects_empty_c0(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_source(Path(tmp))
            path = source / "client_class_histograms.json"
            hist = json.loads(path.read_text())
            hist["client_2"]["num_samples"] = 0
            hist["client_3"]["num_samples"] = 10000
            path.write_text(json.dumps(hist))
            validate_inputs(self.args(source, "federaser"))
            hist["client_0"]["num_samples"] = 0
            hist["client_3"]["num_samples"] = 15000
            path.write_text(json.dumps(hist))
            with self.assertRaisesRegex(ValueError, "Client 0"):
                validate_inputs(self.args(source, "subtraction"))

    def test_check_only_does_not_create_output_or_load_tensor_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_source(Path(tmp))
            output = Path(tmp) / "not_created"
            main(["--source_run", str(source), "--method", "subtraction", "--check_only", "--output_dir", str(output)])
            self.assertFalse(output.exists())

    def test_methods_require_their_own_historical_clients(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_source(Path(tmp))
            (source / "trajectory/round_002/delta_client_00.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                validate_inputs(self.args(source, "subtraction"))
            validate_inputs(self.args(source, "federaser"))
            (source / "trajectory/round_001/delta_client_05.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                validate_inputs(self.args(source, "federaser"))

    def test_kd_rejects_subtraction_from_another_attack_and_edited_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_source(Path(tmp))
            sub = source / "unlearning/subtraction"
            sub.mkdir(parents=True)
            (sub / "final_model.pt").write_bytes(b"subtracted fixture")
            meta = dict(method="subtraction", source_model_sha256="wrong", output_model_sha256=sha256(sub / "final_model.pt"))
            (sub / "run_config.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "exact attacked"):
                validate_inputs(self.args(source, "kd"))
            meta["source_model_sha256"] = sha256(source / "final_model.pt")
            (sub / "run_config.json").write_text(json.dumps(meta))
            validate_inputs(self.args(source, "kd"))
            (sub / "final_model.pt").write_bytes(b"edited")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_inputs(self.args(source, "kd"))

    def test_reference_config_must_match_source_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_source(Path(tmp))
            ref = Path(tmp) / "ref"
            ref.mkdir()
            (ref / "final_model.pt").write_bytes(b"reference")
            config = json.loads((source / "config.json").read_text())
            config["partition_type"] = "noniid"
            (ref / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "partition_type"):
                validate_inputs(self.args(source, "subtraction", "--reference_model", str(ref / "final_model.pt")))

    def test_source_records_must_agree_and_white_patch_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_source(Path(tmp))
            config = json.loads((source / "config.json").read_text())
            config["seed"] = 123
            (source / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "disagree"):
                validate_inputs(self.args(source, "subtraction"))
            config["trigger_type"] = "white_patch"
            (source / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "checkerboard"):
                validate_inputs(self.args(source, "subtraction"))

    def test_subtraction_uses_sample_weight_all_rounds_and_preserves_counters(self):
        original = {"w": Scalar(10), "running_var": Scalar(5), "num_batches_tracked": Scalar(100, False)}
        calls = []
        def load(path, **kwargs):
            calls.append(str(path))
            if Path(path).name == "final_model.pt":
                return original
            value = 1 if "round_001" in str(path) else 2
            return {"w": Scalar(value), "running_var": Scalar(value), "num_batches_tracked": Scalar(10)}
        state = subtract(SimpleNamespace(load=load), Path("source"), [2, 3], 2)
        self.assertAlmostEqual(state["w"].value, 8.8)
        self.assertAlmostEqual(state["running_var"].value, 3.8)
        self.assertEqual(state["num_batches_tracked"].value, 100)
        self.assertEqual(original["w"].value, 10)
        self.assertEqual(len(calls), 3)

    def test_result_report_uses_new_source_and_checkerboard_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_results(path, {"clean_accuracy": 75.26, "triggered_rate": 92.18},
                          {"clean_accuracy": 70, "attack_success_rate": 0},
                          {"source_run": "new_iid_attack", "method": "subtraction", "source_config": {"partition_type": "iid"}})
            self.assertEqual(json.loads((path / "evaluation.json").read_text())["attack_success_rate"], 0)
            report = (path / "report.md").read_text()
            self.assertIn("92.18", report)
            self.assertIn("0.00", report)
            self.assertIn("new_iid_attack", report)

    def test_evaluation_passes_source_checkerboard_to_asr_loader(self):
        clean = SimpleNamespace(dataset=range(10000))
        triggered = SimpleNamespace(dataset=range(9000))
        calls = []
        def asr_loader(*args):
            calls.append(args)
            return triggered
        confusion = [[0] * 10 for _ in range(10)]
        for i in range(10):
            confusion[i][i] = 1000
        confusion[1][0], confusion[1][1] = 100, 900
        def metrics(model, loader, *args):
            if loader is triggered:
                return {"accuracy": 2.5}
            return {"accuracy": 99, "macro_f1": .99, "loss": .01, "per_class": {}, "confusion_matrix": confusion}
        modules = {"src.data.cifar10": SimpleNamespace(get_cifar10_dataloaders=lambda *a, **k: (None, clean)),
                   "src.data.backdoor": SimpleNamespace(create_asr_dataloader=asr_loader),
                   "src.models.small_cnn": SimpleNamespace(SmallCNN=object),
                   "src.training.train_utils": SimpleNamespace(evaluate_extended=metrics)}
        fake_torch = SimpleNamespace(nn=SimpleNamespace(CrossEntropyLoss=object))
        config = dict(batch_size=64, target_class=0, trigger_size=4, trigger_type="checkerboard")
        args = SimpleNamespace(data_dir=Path("data"), reference_model=None)
        with patch.dict(sys.modules, modules):
            result = evaluate(fake_torch, object(), config, args, "cpu")
        self.assertEqual(calls[0][1:], (0, 4, 64, "checkerboard"))
        self.assertEqual(result["attack_success_rate"], 2.5)
        self.assertEqual(result["clean_accuracy"], 99)
        self.assertAlmostEqual(result["untriggered_target_rate"], 100 * 100 / 9000)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is not installed")
class TestReconstructionMath(unittest.TestCase):
    def test_corrected_aggregation_preserves_integer_batchnorm_counters(self):
        import torch
        state = {"weight": torch.tensor([1.]), "running_var": torch.tensor([1.]), "num_batches_tracked": torch.tensor(4)}
        updates = [{"weight": torch.tensor([1.]), "running_var": torch.tensor([.2]), "num_batches_tracked": torch.tensor(7.)}]
        result = aggregate_reconstruction(state, updates, [10])
        self.assertEqual(result["num_batches_tracked"].item(), 4)
        self.assertTrue(torch.allclose(result["weight"], torch.tensor([2.])))
        self.assertTrue(torch.allclose(result["running_var"], torch.tensor([1.2])))


if __name__ == "__main__":
    unittest.main()
