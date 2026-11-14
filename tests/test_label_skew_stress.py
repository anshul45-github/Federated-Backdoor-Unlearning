"""Skew diagnostics, invalid attack splits and matched method routing."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from scripts.run_label_skew_stress import commands, main, partition_report, summarize


class TestLabelSkewStress(unittest.TestCase):
    def config(self):
        return dict(dirichlet_alpha=0.01, seed=42, poison_ratio=0.1, target_class=0)

    def histogram(self):
        # Each client owns one complete class; C0 owns class 1 and C1 class 0.
        return {f"client_{c}": dict(num_samples=5000, class_distribution={str(1-c if c < 2 else c): 5000}) for c in range(10)}

    def test_single_class_skew_and_class_loss_are_reported(self):
        report = partition_report(self.histogram(), self.config())
        self.assertEqual(report["poisoned_samples"], 500)
        self.assertEqual(report["malicious_client_weight"], .1)
        self.assertEqual(report["classes_lost_on_removal"], [1])
        self.assertEqual(report["clients"][0]["label_entropy_nats"], 0)
        self.assertEqual(report["clients"][0]["dominant_class_fraction"], 1)
        self.assertFalse(report["errors"])

    def test_empty_retained_client_is_reported_without_redistribution(self):
        hist = self.histogram()
        hist["client_2"] = dict(num_samples=0, class_distribution={})
        hist["client_3"] = dict(num_samples=10000, class_distribution={"2": 5000, "3": 5000})
        report = partition_report(hist, self.config())
        self.assertEqual(report["empty_clients"], [2])
        self.assertIsNone(report["clients"][2]["dominant_class_fraction"])
        self.assertFalse(report["errors"])

    def test_target_only_malicious_client_cannot_host_attack(self):
        hist = self.histogram()
        hist["client_0"], hist["client_1"] = hist["client_1"], hist["client_0"]
        self.assertTrue(partition_report(hist, self.config())["errors"])

    def test_insufficient_poison_candidates_are_not_silently_capped(self):
        hist = self.histogram()
        hist["client_0"]["class_distribution"] = {"0": 4999, "1": 1}
        hist["client_1"]["class_distribution"] = {"0": 1, "1": 4999}
        self.assertTrue(partition_report(hist, self.config())["errors"])

    def test_incomplete_histogram_is_rejected(self):
        hist = self.histogram()
        hist["client_0"]["num_samples"] = 4999
        with self.assertRaises(ValueError):
            partition_report(hist, self.config())

    def test_methods_share_source_and_kd_uses_new_subtraction(self):
        run = Path("isolated/seed_42")
        stages = commands(run)
        self.assertEqual([s for s, _ in stages], ["attack", "subtraction", "kd", "federaser"])
        self.assertIn("--log_trajectory", stages[0][1])
        for _, command in stages[1:]:
            self.assertEqual(command[command.index("--source_run")+1], str(run / "attack"))
        kd = stages[2][1]
        self.assertEqual(kd[kd.index("--subtraction_dir")+1], str(run / "attack/unlearning/subtraction"))

    def test_summary_reports_changes_against_new_attack(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            source = run / "attack"
            source.mkdir()
            (source / "final_results.json").write_text(json.dumps(dict(clean_accuracy=50, triggered_rate=90,
                untriggered_airplane_rate=10, clean_macro_f1=.4)))
            for method in ["subtraction", "kd", "federaser_Ecal2"]:
                out = source / "unlearning" / method
                out.mkdir(parents=True)
                (out / "evaluation.json").write_text(json.dumps(dict(clean_accuracy=30, attack_success_rate=2,
                    untriggered_target_rate=1, macro_f1_pct=25)))
            summarize(run)
            with (run / "comparison.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(float(rows[1]["ca_change_pp"]), -20)
            self.assertEqual(float(rows[1]["asr_change_pp"]), -88)
            self.assertIn("collapsed CA", (run / "report.md").read_text())

    def test_plan_creates_no_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "unused"
            main(["--plan", "--output_dir", str(output)])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
