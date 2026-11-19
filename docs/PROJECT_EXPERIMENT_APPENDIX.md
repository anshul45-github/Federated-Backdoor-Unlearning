# Experiment appendix

Companion to the [project experiment review](PROJECT_EXPERIMENT_REVIEW.md). CA, ASR, UTR, F1, and CF agreement are percentages. A dash denotes an unavailable or inapplicable value. Linked CSVs and JSON preserve source precision. Matched onset-free data were read from commit f81d1fd.

[Historical and white-patch benchmark](#historical-and-white-patch-benchmark) · [Checkerboard runs and component patching](#checkerboard-runs-and-component-patching) · [TCUG continuations and repairs](#tcug-continuations-and-repairs) · [Onset-free sweeps](#onset-free-sweeps) · [Onset-free diagnostics](#onset-free-diagnostics)

## Historical and white-patch benchmark

### Stage 7A superseded evaluations

The original 7A values are reproduced for traceability. The 7B audit found ineffective client exclusion, proxy overlap, test-driven checkpoint selection, and a BatchNorm geometry error. The corrected 7B repair values and final Stage 7C factorial appear in the main review. [7A source](historical/7a/stage_7a_final_report.md) · [7B corrections](historical/7b/stage_7b_corrected_report.md)

| Regime | Method | CA | ASR | Original CF agreement | Original CF gap |
| --- | --- | --- | --- | --- | --- |
| C1 | Subtraction | 10.30% | 0.00% | 10.01% | 89.99% |
| C1 | Subtraction+KD | 57.61% | 3.53% | 66.63% | 33.37% |
| C1 | FedEraser | 74.78% | 2.58% | 89.35% | 10.65% |
| C2 | Subtraction | 11.52% | 0.00% | 12.08% | 87.92% |
| C2 | Subtraction+KD | 50.27% | 4.89% | 58.10% | 41.90% |
| C2 | FedEraser | 73.25% | 2.24% | 84.74% | 15.26% |
| C3 | Subtraction | 11.33% | 0.00% | 17.45% | 82.55% |
| C3 | Subtraction+KD | 32.00% | 12.99% | 45.26% | 54.74% |
| C3 | FedEraser | 74.73% | 2.32% | 67.92% | 32.08% |
| C4 | Subtraction | 10.25% | 0.00% | 13.15% | 86.85% |
| C4 | Subtraction+KD | 18.67% | 0.29% | 26.90% | 73.10% |
| C4 | FedEraser | 74.82% | 2.28% | 67.10% | 32.90% |

Archived prototype scope, run identifiers, and evaluations are indexed by the [historical benchmark](baseline/canonical_benchmark.md), [Stage 6C](historical/6c/stage_6c_analysis.md), [Stage 6D Phase A](historical/6d/stage_6d_phase_a_report.md), [nine Phase B runs](historical/6d/stage_6d_phase_b_report.md), [Stage 7B](historical/7b/stage_7b_corrected_report.md), and [final Stage 7C audit](historical/7c/stage7c_final_factorial_robustness_audit.md). The historical storage audit [lists interrupted, invalidated and archived trajectories](historical/storage/storage_archive_verification.md).

### All ten saved 50-round FedEraser endpoints

| Run | Status | CA | ASR | Recorded UTR | Clean CF | Triggered CF | Log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [iid](../results/phase2/federaser/iid/eval_results.json) | Superseded, weight decay and UTR | 66.13 | 3.50 | 3.77 | 78.87 | 78.62 | [rounds](../results/phase2/federaser/iid/reconstruction_logs.csv) |
| [iid_Ecal1](../results/phase2/federaser/iid_Ecal1/eval_results.json) | Superseded, seed and BN counters | 72.43 | 3.18 | 3.63 | 86.08 | 85.68 | [rounds](../results/phase2/federaser/iid_Ecal1/reconstruction_logs.csv) |
| [iid_Ecal2](../results/phase2/federaser/iid_Ecal2/eval_results.json) | Superseded, seed and BN counters | 74.96 | 2.18 | 2.51 | 90.57 | 90.83 | [rounds](../results/phase2/federaser/iid_Ecal2/reconstruction_logs.csv) |
| [iid_Ecal1_v2](../results/phase2/federaser/iid_Ecal1_v2/eval_results.json) | Corrected | 72.11 | 2.67 | 2.98 | 84.95 | 84.61 | [rounds](../results/phase2/federaser/iid_Ecal1_v2/reconstruction_logs.csv) |
| [iid_Ecal2_v2](../results/phase2/federaser/iid_Ecal2_v2/eval_results.json) | Corrected | 74.99 | 2.34 | 2.49 | 89.29 | 88.79 | [rounds](../results/phase2/federaser/iid_Ecal2_v2/reconstruction_logs.csv) |
| [noniid](../results/phase2/federaser/noniid/eval_results.json) | Superseded, weight decay and UTR | 58.67 | 3.73 | 4.28 | 76.35 | 75.61 | [rounds](../results/phase2/federaser/noniid/reconstruction_logs.csv) |
| [noniid_Ecal1](../results/phase2/federaser/noniid_Ecal1/eval_results.json) | Superseded, seed and BN counters | 65.40 | 4.12 | 5.08 | 84.57 | 83.77 | [rounds](../results/phase2/federaser/noniid_Ecal1/reconstruction_logs.csv) |
| [noniid_Ecal2](../results/phase2/federaser/noniid_Ecal2/eval_results.json) | Superseded, seed and BN counters | 69.00 | 2.54 | 3.14 | 87.92 | 86.94 | [rounds](../results/phase2/federaser/noniid_Ecal2/reconstruction_logs.csv) |
| [noniid_Ecal1_v2](../results/phase2/federaser/noniid_Ecal1_v2/eval_results.json) | Corrected | 65.75 | 3.53 | 4.13 | 85.34 | 84.61 | [rounds](../results/phase2/federaser/noniid_Ecal1_v2/reconstruction_logs.csv) |
| [noniid_Ecal2_v2](../results/phase2/federaser/noniid_Ecal2_v2/eval_results.json) | Corrected | 68.89 | 3.09 | 4.11 | 88.18 | 86.82 | [rounds](../results/phase2/federaser/noniid_Ecal2_v2/reconstruction_logs.csv) |

For other Phase 1/2 endpoints, the [corrected master table](../results/phase2/federaser/corrected_evaluation_v2/federaser_comparison.csv) gives loss, CA, ASR, UTR, macro F1, clean and triggered CF agreement, target recall and timer scope. The [original evaluation](../results/phase2/phase2_evaluation_superseded.json), [subsequent corrected evaluation](../results/phase2/phase2_evaluation_corrected.json), [four superseded Ecal configurations](../results/phase2/federaser/4_configs/federaser_extended_comparison.csv), and [controlled timing microbenchmark](../results/phase2/federaser/controlled_timing.json) preserve the underlying audit trail.

### White-patch source runs and class-level ASR

| Folder | Last logged round | Last logged CA | Last logged triggered rate | Source |
| --- | --- | --- | --- | --- |
| phase1/iid_attack | 50 | 75.01 | 56.07 | [all rounds](../results/phase1/iid_attack/metrics_per_round.csv) |
| phase1/iid_clean | 50 | 75.11 | 2.73 | [all rounds](../results/phase1/iid_clean/metrics_per_round.csv) |
| phase1/noniid_attack | 50 | 69.80 | 32.21 | [all rounds](../results/phase1/noniid_attack/metrics_per_round.csv) |
| phase1/noniid_clean | 50 | 70.10 | 3.30 | [all rounds](../results/phase1/noniid_clean/metrics_per_round.csv) |
| phase2/iid_attack_logged | 50 | 75.01 | 56.07 | [all rounds](../results/phase2/iid_attack_logged/metrics_per_round.csv) |
| phase2/iid_counterfactual | 50 | 74.87 | 3.01 | [all rounds](../results/phase2/iid_counterfactual/metrics_per_round.csv) |
| phase2/noniid_attack_logged | 50 | 69.80 | 32.21 | [all rounds](../results/phase2/noniid_attack_logged/metrics_per_round.csv) |
| phase2/noniid_counterfactual | 50 | 69.64 | 3.40 | [all rounds](../results/phase2/noniid_counterfactual/metrics_per_round.csv) |

Round-log metrics are read directly from the named trajectories. The corrected endpoint comparison is the reference for the main tables.

| Phase 1 model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IID Clean | 1.7 | 5.6 | 2.1 | 1.8 | 0.9 | 0.6 | 0.9 | 7.1 | 3.9 |
| IID Attack | 51.7 | 65.0 | 59.9 | 63.7 | 50.9 | 35.9 | 49.2 | 78.8 | 49.5 |
| Non-IID Clean | 3.1 | 7.0 | 2.1 | 3.2 | 1.6 | 0.8 | 1.6 | 7.6 | 2.7 |
| Non-IID Attack | 35.1 | 35.6 | 33.1 | 36.9 | 21.4 | 11.9 | 38.6 | 45.6 | 31.7 |

Class IDs run from automobile (1) to truck (9). Airplane is the target and is excluded. [Phase 1 per-class data](../results/phase1/per_class_asr.csv) · [Smoke result](../results/smoke/smoke_test_dir/final_results.json)

## Checkerboard runs and component patching

| Run | Endpoint | Status | CA | ASR | Source |
| --- | --- | --- | --- | --- | --- |
| [iid_20261024_015516](../results/checkerboard/iid_20261024_015516/config.json) | Initial attack attempt | Config only | — | — | — |
| [iid_20261024_015516_rerun_001](../results/checkerboard/iid_20261024_015516_rerun_001/final_results.json) | Attacked model | Completed | 75.26 | 92.18 | [rounds](../results/checkerboard/iid_20261024_015516_rerun_001/metrics_per_round.csv) |
| [iid_20261024_015516_rerun_001](../results/checkerboard/iid_20261024_015516_rerun_001/unlearning/20261024_205805_550/federaser_Ecal2/results.csv) | federaser | Completed | 75.19 | 1.18 | [extended metrics](../results/checkerboard/iid_20261024_015516_rerun_001/unlearning/20261024_205805_550/federaser_Ecal2/evaluation.json) |
| [iid_20261024_015516_rerun_001](../results/checkerboard/iid_20261024_015516_rerun_001/unlearning/20261024_205805_550/kd/results.csv) | kd | Completed | 73.09 | 0.41 | [extended metrics](../results/checkerboard/iid_20261024_015516_rerun_001/unlearning/20261024_205805_550/kd/evaluation.json) |
| [iid_20261024_015516_rerun_001](../results/checkerboard/iid_20261024_015516_rerun_001/unlearning/20261024_205805_550/subtraction/results.csv) | subtraction | Completed | 68.21 | 0.02 | [extended metrics](../results/checkerboard/iid_20261024_015516_rerun_001/unlearning/20261024_205805_550/subtraction/evaluation.json) |
| [noniid_20261024_150813_189](../results/checkerboard/noniid_20261024_150813_189/final_results.json) | Attacked model | Completed | 71.14 | 80.89 | [rounds](../results/checkerboard/noniid_20261024_150813_189/metrics_per_round.csv) |
| [noniid_20261024_150813_189](../results/checkerboard/noniid_20261024_150813_189/unlearning/20261024_205805_550/federaser_Ecal2/results.csv) | federaser | Completed | 70.18 | 0.94 | [extended metrics](../results/checkerboard/noniid_20261024_150813_189/unlearning/20261024_205805_550/federaser_Ecal2/evaluation.json) |
| [noniid_20261024_150813_189](../results/checkerboard/noniid_20261024_150813_189/unlearning/20261024_205805_550/kd/results.csv) | kd | Completed | 59.35 | 0.79 | [extended metrics](../results/checkerboard/noniid_20261024_150813_189/unlearning/20261024_205805_550/kd/evaluation.json) |
| [noniid_20261024_150813_189](../results/checkerboard/noniid_20261024_150813_189/unlearning/20261024_205805_550/subtraction/results.csv) | subtraction | Completed | 31.25 | 0.13 | [extended metrics](../results/checkerboard/noniid_20261024_150813_189/unlearning/20261024_205805_550/subtraction/evaluation.json) |
| [alpha001_20261029_214800_489874](../results/label_skew/alpha001_20261029_214800_489874/seed_42/partition_report.json) | Skew partition preflight | Partition only | — | — | — |
| [alpha001_20261029_214828_723364](../results/label_skew/alpha001_20261029_214828_723364/seed_42/comparison.csv) | attack | Completed | 33.48 | 99.97 | [details](../results/label_skew/alpha001_20261029_214828_723364/seed_42/attack/metrics_per_round.csv) |
| [alpha001_20261029_214828_723364](../results/label_skew/alpha001_20261029_214828_723364/seed_42/comparison.csv) | subtraction | Completed | 10.00 | 100.00 | [details](../results/label_skew/alpha001_20261029_214828_723364/seed_42/attack/unlearning/subtraction/evaluation.json) |
| [alpha001_20261029_214828_723364](../results/label_skew/alpha001_20261029_214828_723364/seed_42/comparison.csv) | kd | Completed | 10.00 | 100.00 | [details](../results/label_skew/alpha001_20261029_214828_723364/seed_42/attack/unlearning/kd/evaluation.json) |
| [alpha001_20261029_214828_723364](../results/label_skew/alpha001_20261029_214828_723364/seed_42/comparison.csv) | federaser_Ecal2 | Completed | 34.86 | 6.57 | [details](../results/label_skew/alpha001_20261029_214828_723364/seed_42/attack/unlearning/federaser_Ecal2/evaluation.json) |

### Stage 1, all-channel component patch screen

| Donor round | Blend | CA | ASR | CA change (pp) | ASR change (pp) |
| --- | --- | --- | --- | --- | --- |
| 10 | 0.25 | 73.76 | 79.42 | -1.38 | -12.40 |
| 10 | 0.5 | 71.08 | 55.24 | -4.06 | -36.58 |
| 10 | 0.75 | 66.38 | 35.49 | -8.76 | -56.33 |
| 10 | 1.0 | 59.92 | 26.18 | -15.22 | -65.64 |
| 20 | 0.25 | 74.90 | 86.98 | -0.24 | -4.84 |
| 20 | 0.5 | 73.68 | 78.27 | -1.46 | -13.56 |
| 20 | 0.75 | 71.54 | 63.42 | -3.60 | -28.40 |
| 20 | 1.0 | 68.90 | 48.93 | -6.24 | -42.89 |
| 29 | 0.25 | 75.18 | 89.00 | 0.04 | -2.82 |
| 29 | 0.5 | 74.76 | 83.67 | -0.38 | -8.16 |
| 29 | 0.75 | 73.92 | 75.47 | -1.22 | -16.36 |
| 29 | 1.0 | 72.68 | 64.98 | -2.46 | -26.84 |

The Stage 1 score divides ASR reduction by CA loss clamped at a small positive constant. The chosen donor/blend was G29 at 0.25. [Full screen and scores](../results/checkerboard/iid_20261024_015516_rerun_001/component_patching/stage1.csv)

### Stage 2, channel and order ablations

| Order | Best prefix k | CA | ASR | CA within 1 pp budget |
| --- | --- | --- | --- | --- |
| single_channel_score | 42 | 75.20 | 88.58 | Yes |
| weight_drift | 64 | 75.18 | 89.00 | Yes |
| random_00 | 62 | 75.02 | 88.84 | Yes |
| random_01 | 64 | 75.18 | 89.00 | Yes |
| random_02 | 63 | 75.18 | 89.00 | Yes |
| random_03 | 62 | 75.24 | 89.00 | Yes |
| random_04 | 61 | 75.16 | 88.87 | Yes |
| random_05 | 64 | 75.18 | 89.00 | Yes |
| random_06 | 64 | 75.18 | 89.00 | Yes |
| random_07 | 63 | 75.18 | 89.00 | Yes |
| random_08 | 63 | 75.22 | 89.00 | Yes |
| random_09 | 64 | 75.18 | 89.00 | Yes |

All 64 single-channel trials and all 780 prefix evaluations, including k = 0 and 64 for each order, are saved in the original result tables. The selected IID patch used a single-channel-score prefix with 42 channels. The manifest records zero optional calibration steps. [64 individual trials](../results/checkerboard/iid_20261024_015516_rerun_001/component_patching/single_channel_scores.csv) · [780 prefix trials](../results/checkerboard/iid_20261024_015516_rerun_001/component_patching/channel_curves.csv) · [All orders](../results/checkerboard/iid_20261024_015516_rerun_001/component_patching/ranking_orders.json) · [Held-out metrics](../results/checkerboard/iid_20261024_015516_rerun_001/component_patching/held_out_results.json) · [Search settings](../results/checkerboard/iid_20261024_015516_rerun_001/component_patching/manifest.json)

## TCUG continuations and repairs

Each row is a saved outcome or last checkpoint in an incomplete run. Comparison CSVs also contain the baselines reevaluated for each run. TCUG-0 through TCUG-2 continuation endpoints are round 50. Healing endpoints are round 53. Direct edits start from round 50.

| Experiment | Run / partition | Evaluated endpoint | CA | ASR | UTR | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| CPU TCUG-0 | [iid](../results/rollback/iid/comparison.csv) | rollback_21_honest_rounds | 75.17 | 1.34 | 3.39 | 75.01 |
| CPU TCUG-0 | [noniid](../results/rollback/noniid/comparison.csv) | rollback_21_honest_rounds | 69.97 | 2.09 | 4.51 | 68.68 |
| GPU continuation | [20261115_015724/tcug-0/iid](../results/tcug_gpu/20261115_015724/tcug-0/iid/comparison.csv) | rollback_21_honest_rounds | 75.14 | 1.53 | 3.46 | 74.95 |
| GPU continuation | [20261115_015724/tcug-0/noniid](../results/tcug_gpu/20261115_015724/tcug-0/noniid/comparison.csv) | rollback_21_honest_rounds | 70.01 | 1.83 | 4.98 | 68.83 |
| GPU continuation | [20261115_015724/tcug-1/iid](../results/tcug_gpu/20261115_015724/tcug-1/iid/metrics_per_round.csv) | TCUG-1, incomplete at round 46 | 74.57 | 1.37 | 2.89 | — |
| GPU continuation | [20261115_120004/tcug-0/iid](../results/tcug_gpu/20261115_120004/tcug-0/iid/comparison.csv) | rollback_21_honest_rounds | 75.43 | 1.44 | 3.51 | 75.26 |
| GPU continuation | [20261115_120004/tcug-0/noniid](../results/tcug_gpu/20261115_120004/tcug-0/noniid/comparison.csv) | rollback_21_honest_rounds | 70.27 | 1.89 | 4.82 | 69.06 |
| GPU continuation | [20261115_120004/tcug-1/iid](../results/tcug_gpu/20261115_120004/tcug-1/iid/comparison.csv) | tcug_1 | 74.99 | 1.62 | 3.89 | 74.79 |
| GPU continuation | [20261115_120004/tcug-1/noniid](../results/tcug_gpu/20261115_120004/tcug-1/noniid/comparison.csv) | tcug_1 | 70.93 | 1.87 | 4.70 | 69.89 |
| GPU continuation | [tcug2_20261115_125159/tcug-2/iid](../results/tcug_gpu/tcug2_20261115_125159/tcug-2/iid/comparison.csv) | tcug_2 | 75.09 | 1.49 | 3.69 | 74.86 |
| GPU continuation | [tcug2_20261115_125159/tcug-2/noniid](../results/tcug_gpu/tcug2_20261115_125159/tcug-2/noniid/comparison.csv) | tcug_2 | 70.59 | 1.77 | 4.78 | 69.54 |
| Fixed mean | [iid](../results/training_free_tcug/iid/results.csv) | training_free_tcug | 72.86 | 0.09 | — | — |
| Fixed mean | [noniid](../results/training_free_tcug/noniid/results.csv) | training_free_tcug | 69.67 | 0.12 | — | — |
| Top-three fixed donor window | [iid](../results/training_free_tcug_2/iid/results.csv) | training_free_tcug_2 | 71.92 | 0.04 | — | — |
| Top-three fixed donor window | [noniid](../results/training_free_tcug_2/noniid/results.csv) | training_free_tcug_2 | 68.52 | 0.11 | — | — |
| Rolling top-three | [iid](../results/training_free_rolling_tcug_2/iid/results.csv) | training_free_rolling_tcug_2 | 50.68 | 0.03 | — | — |
| Rolling top-three | [noniid](../results/training_free_rolling_tcug_2/noniid/results.csv) | training_free_rolling_tcug_2 | 40.86 | 0.24 | — | — |
| Top-three + healing | [20261115_144254/iid](../results/tcug_followup_gpu/20261115_144254/iid/comparison.csv) | training_free_tcug_2 | 71.91 | 0.04 | 1.40 | 71.67 |
| Top-three + healing | [20261115_144254/iid](../results/tcug_followup_gpu/20261115_144254/iid/comparison.csv) | training_free_tcug_2_plus_3_rounds | 75.57 | 0.16 | 2.90 | 75.37 |
| Top-three + healing | [20261115_144254/noniid](../results/tcug_followup_gpu/20261115_144254/noniid/comparison.csv) | training_free_tcug_2 | 68.52 | 0.11 | 3.08 | 67.19 |
| Top-three + healing | [20261115_144254/noniid](../results/tcug_followup_gpu/20261115_144254/noniid/comparison.csv) | training_free_tcug_2_plus_3_rounds | 70.70 | 0.24 | 3.98 | 69.72 |
| Projection sweep | [iid](../results/direction_preserving_tcug/iid/comparison.csv) | lambda_0 | 65.21 | 0.00 | 0.66 | 65.36 |
| Projection sweep | [iid](../results/direction_preserving_tcug/iid/comparison.csv) | lambda_0.25 | 65.28 | 0.00 | 0.60 | 65.36 |
| Projection sweep | [iid](../results/direction_preserving_tcug/iid/comparison.csv) | lambda_0.5 | 65.43 | 0.00 | 0.54 | 65.46 |
| Projection sweep | [iid](../results/direction_preserving_tcug/iid/comparison.csv) | lambda_1 | 65.57 | 0.00 | 0.47 | 65.46 |
| Projection sweep | [noniid](../results/direction_preserving_tcug/noniid/comparison.csv) | lambda_0 | 27.54 | 0.00 | 0.44 | 14.78 |
| Projection sweep | [noniid](../results/direction_preserving_tcug/noniid/comparison.csv) | lambda_0.25 | 27.53 | 0.00 | 0.43 | 14.82 |
| Projection sweep | [noniid](../results/direction_preserving_tcug/noniid/comparison.csv) | lambda_0.5 | 27.52 | 0.01 | 0.44 | 14.82 |
| Projection sweep | [noniid](../results/direction_preserving_tcug/noniid/comparison.csv) | lambda_1 | 27.52 | 0.01 | 0.44 | 14.94 |
| Projection λ = 1 + healing | [20261116_184136/iid](../results/direction_preserving_followup_gpu/20261116_184136/iid/comparison.csv) | direction_preserving_tcug_lambda_1_plus_3_rounds | 75.19 | 0.18 | 2.84 | 75.05 |
| Projection λ = 1 + healing | [20261116_184136/noniid](../results/direction_preserving_followup_gpu/20261116_184136/noniid/comparison.csv) | direction_preserving_tcug_lambda_1_plus_3_rounds | 67.58 | 0.50 | 4.70 | 66.30 |
| Projection λ = 0 + healing | [20261116_184856/iid](../results/direction_preserving_lambda0_followup_gpu/20261116_184856/iid/comparison.csv) | direction_preserving_tcug_lambda_0_plus_3_rounds | 75.33 | 0.16 | 2.68 | 75.19 |
| Projection λ = 0 + healing | [20261116_184856/noniid](../results/direction_preserving_lambda0_followup_gpu/20261116_184856/noniid/comparison.csv) | direction_preserving_tcug_lambda_0_plus_3_rounds | 67.20 | 0.46 | 4.66 | 66.01 |
| CBSH + healing | [20261116_193407/iid](../results/cbsh_gpu/20261116_193407/iid/comparison.csv) | direction_preserving_tcug_lambda_0_plus_cbsh_3_rounds | 74.41 | 0.27 | 3.00 | 74.21 |
| CBSH + healing | [20261116_193407/noniid](../results/cbsh_gpu/20261116_193407/noniid/comparison.csv) | direction_preserving_tcug_lambda_0_plus_cbsh_3_rounds | 63.88 | 0.22 | 3.68 | 61.85 |
| Random five-client healing | [seed42/iid](../results/cbsh_random_gpu/seed42/iid/comparison.csv) | direction_preserving_tcug_lambda_0_plus_random_healing_3_rounds | 75.21 | 0.20 | 2.56 | 74.95 |
| Random five-client healing | [seed42/noniid](../results/cbsh_random_gpu/seed42/noniid/comparison.csv) | direction_preserving_tcug_lambda_0_plus_random_healing_3_rounds | 64.94 | 0.37 | 3.73 | 61.38 |

### Continuation and healing round histories

| Experiment | Run / partition | Logged rounds | Metadata |
| --- | --- | --- | --- |
| CPU TCUG-0 | [iid](../results/rollback/iid/metrics_per_round.csv) | 30–50 | [config](../results/rollback/iid/run_config.json) |
| CPU TCUG-0 | [noniid](../results/rollback/noniid/metrics_per_round.csv) | 30–50 | [config](../results/rollback/noniid/run_config.json) |
| GPU continuation | [20261115_015724/tcug-0/iid](../results/tcug_gpu/20261115_015724/tcug-0/iid/metrics_per_round.csv) | 30–50 | [config](../results/tcug_gpu/20261115_015724/tcug-0/iid/run_config.json) |
| GPU continuation | [20261115_015724/tcug-0/noniid](../results/tcug_gpu/20261115_015724/tcug-0/noniid/metrics_per_round.csv) | 30–50 | [config](../results/tcug_gpu/20261115_015724/tcug-0/noniid/run_config.json) |
| GPU continuation | [20261115_015724/tcug-1/iid](../results/tcug_gpu/20261115_015724/tcug-1/iid/metrics_per_round.csv) | 30–46 | [config](../results/tcug_gpu/20261115_015724/tcug-1/iid/run_config.json) |
| GPU continuation | [20261115_120004/tcug-0/iid](../results/tcug_gpu/20261115_120004/tcug-0/iid/metrics_per_round.csv) | 30–50 | [config](../results/tcug_gpu/20261115_120004/tcug-0/iid/run_config.json) |
| GPU continuation | [20261115_120004/tcug-0/noniid](../results/tcug_gpu/20261115_120004/tcug-0/noniid/metrics_per_round.csv) | 30–50 | [config](../results/tcug_gpu/20261115_120004/tcug-0/noniid/run_config.json) |
| GPU continuation | [20261115_120004/tcug-1/iid](../results/tcug_gpu/20261115_120004/tcug-1/iid/metrics_per_round.csv) | 30–50 | [config](../results/tcug_gpu/20261115_120004/tcug-1/iid/run_config.json) |
| GPU continuation | [20261115_120004/tcug-1/noniid](../results/tcug_gpu/20261115_120004/tcug-1/noniid/metrics_per_round.csv) | 30–50 | [config](../results/tcug_gpu/20261115_120004/tcug-1/noniid/run_config.json) |
| GPU continuation | [tcug2_20261115_125159/tcug-2/iid](../results/tcug_gpu/tcug2_20261115_125159/tcug-2/iid/metrics_per_round.csv) | 30–50 | [config](../results/tcug_gpu/tcug2_20261115_125159/tcug-2/iid/run_config.json), [graft history](../results/tcug_gpu/tcug2_20261115_125159/tcug-2/iid/graft_history.json) |
| GPU continuation | [tcug2_20261115_125159/tcug-2/noniid](../results/tcug_gpu/tcug2_20261115_125159/tcug-2/noniid/metrics_per_round.csv) | 30–50 | [config](../results/tcug_gpu/tcug2_20261115_125159/tcug-2/noniid/run_config.json), [graft history](../results/tcug_gpu/tcug2_20261115_125159/tcug-2/noniid/graft_history.json) |
| Top-three + healing | [20261115_144254/iid](../results/tcug_followup_gpu/20261115_144254/iid/metrics_per_round.csv) | 51–53 | [config](../results/tcug_followup_gpu/20261115_144254/iid/run_config.json) |
| Top-three + healing | [20261115_144254/noniid](../results/tcug_followup_gpu/20261115_144254/noniid/metrics_per_round.csv) | 51–53 | [config](../results/tcug_followup_gpu/20261115_144254/noniid/run_config.json) |
| Projection λ = 1 + healing | [20261116_184136/iid](../results/direction_preserving_followup_gpu/20261116_184136/iid/metrics_per_round.csv) | 51–53 | [config](../results/direction_preserving_followup_gpu/20261116_184136/iid/run_config.json) |
| Projection λ = 1 + healing | [20261116_184136/noniid](../results/direction_preserving_followup_gpu/20261116_184136/noniid/metrics_per_round.csv) | 51–53 | [config](../results/direction_preserving_followup_gpu/20261116_184136/noniid/run_config.json) |
| Projection λ = 0 + healing | [20261116_184856/iid](../results/direction_preserving_lambda0_followup_gpu/20261116_184856/iid/metrics_per_round.csv) | 51–53 | [config](../results/direction_preserving_lambda0_followup_gpu/20261116_184856/iid/run_config.json) |
| Projection λ = 0 + healing | [20261116_184856/noniid](../results/direction_preserving_lambda0_followup_gpu/20261116_184856/noniid/metrics_per_round.csv) | 51–53 | [config](../results/direction_preserving_lambda0_followup_gpu/20261116_184856/noniid/run_config.json) |
| CBSH + healing | [20261116_193407/iid](../results/cbsh_gpu/20261116_193407/iid/metrics_per_round.csv) | 51–53 | [config](../results/cbsh_gpu/20261116_193407/iid/run_config.json), [client selection](../results/cbsh_gpu/20261116_193407/iid/selection_history.json) |
| CBSH + healing | [20261116_193407/noniid](../results/cbsh_gpu/20261116_193407/noniid/metrics_per_round.csv) | 51–53 | [config](../results/cbsh_gpu/20261116_193407/noniid/run_config.json), [client selection](../results/cbsh_gpu/20261116_193407/noniid/selection_history.json) |
| Random five-client healing | [seed42/iid](../results/cbsh_random_gpu/seed42/iid/metrics_per_round.csv) | 51–53 | [config](../results/cbsh_random_gpu/seed42/iid/run_config.json), [client selection](../results/cbsh_random_gpu/seed42/iid/selection_history.json) |
| Random five-client healing | [seed42/noniid](../results/cbsh_random_gpu/seed42/noniid/metrics_per_round.csv) | 51–53 | [config](../results/cbsh_random_gpu/seed42/noniid/run_config.json), [client selection](../results/cbsh_random_gpu/seed42/noniid/selection_history.json) |

The direct-edit [projection histories](../results/direction_preserving_tcug/) record sign, projection scale and cap behavior per round. The rolling and fixed-window [reconstruction histories](../results/training_free_rolling_tcug_2/) give donor choices and clipping. In the direct fixed-mean and top-three tables, extended metrics are available in adjacent evaluation JSON files.

## Onset-free sweeps

[Compact results JSON](https://github.com/anshul45-github/Federated-Backdoor-Unlearning/blob/f81d1fdfa3d10630d4438cd8df6a600dc51e3c4c/results/onsetfree/summary_metrics.json) · [Measured report](https://github.com/anshul45-github/Federated-Backdoor-Unlearning/blob/f81d1fdfa3d10630d4438cd8df6a600dc51e3c4c/results/onsetfree/REPORT.md). Table cells are CA / ASR. The same method row in the two partition columns represents two independent evaluations. The two partition labels each refer to one attack trajectory.

### Matched references and default L0 controls

| Reference | IID | Non-IID | IID wall seconds | Non-IID wall seconds |
| --- | --- | --- | --- | --- |
| attack | 74.70 / 51.93 | 71.59 / 29.32 | 2083.33 | 2091.06 |
| counterfactual | 74.92 / 3.08 | 70.32 / 3.41 | 1454.83 | 1515.35 |
| federaser_ecal1 | 71.99 / 2.33 | 67.69 / 2.51 | 726.32 | 748.58 |
| federaser_ecal2 | 75.15 / 2.80 | 69.95 / 2.59 | 1409.39 | 1473.97 |

| Default-save model | IID | Non-IID |
| --- | --- | --- |
| c1 | 64.31 / 0.11 | 27.15 / 1.06 |
| c2 | 64.46 / 0.08 | 26.69 / 1.06 |
| c3_random | 65.37 / 0.09 | 29.58 / 2.23 |
| c4_malicious | 74.97 / 45.20 | 70.13 / 29.68 |
| l0 | 67.63 / 0.24 | 34.30 / 0.69 |
| l0_bn_recalibrated | 67.53 / 0.41 | 43.23 / 0.74 |

### L0 rank, scale, and random seeds

| Method | Rank | IID scale 0.5 | IID scale 1 | Non-IID scale 0.5 | Non-IID scale 1 |
| --- | --- | --- | --- | --- | --- |
| C1 parameter subtraction | — | 72.63 / 1.29 | 64.31 / 0.11 | 39.62 / 3.67 | 27.15 / 1.06 |
| C2 contrast | — | 72.44 / 1.12 | 64.46 / 0.08 | 39.32 / 3.51 | 26.69 / 1.06 |
| C3 random, seed 42 | 4 | 72.67 / 1.19 | 64.87 / 0.12 | 39.93 / 3.42 | 26.96 / 1.13 |
| C3 random, seed 42 | 8 | 72.41 / 1.12 | 64.06 / 0.07 | 40.75 / 3.71 | 28.17 / 1.70 |
| C3 random, seed 42 | 16 | 72.80 / 1.14 | 64.63 / 0.09 | 43.56 / 3.66 | 30.34 / 1.89 |
| C3 random, seed 42 | 32 | 72.88 / 1.28 | 65.37 / 0.09 | 43.20 / 3.94 | 29.58 / 2.23 |
| C3 random, seed 42 | 64 | 72.87 / 1.27 | 66.02 / 0.09 | 47.77 / 3.71 | 35.29 / 2.21 |
| C3 random, seed 43 | 4 | 72.60 / 1.18 | 64.68 / 0.09 | 39.53 / 3.78 | 26.26 / 1.29 |
| C3 random, seed 43 | 8 | 72.68 / 1.20 | 64.54 / 0.08 | 40.59 / 3.49 | 27.06 / 1.33 |
| C3 random, seed 43 | 16 | 72.95 / 1.22 | 65.58 / 0.09 | 42.26 / 3.22 | 28.47 / 1.21 |
| C3 random, seed 43 | 32 | 72.68 / 1.22 | 65.33 / 0.07 | 43.49 / 4.01 | 30.11 / 1.86 |
| C3 random, seed 43 | 64 | 72.89 / 1.27 | 66.04 / 0.10 | 47.90 / 3.74 | 35.66 / 2.46 |
| C3 random, seed 44 | 4 | 72.58 / 1.17 | 64.87 / 0.10 | 40.44 / 3.29 | 27.17 / 1.02 |
| C3 random, seed 44 | 8 | 72.89 / 1.23 | 64.95 / 0.10 | 41.34 / 3.67 | 27.42 / 1.34 |
| C3 random, seed 44 | 16 | 72.60 / 1.12 | 64.70 / 0.08 | 41.63 / 3.90 | 28.42 / 1.58 |
| C3 random, seed 44 | 32 | 72.94 / 1.27 | 65.66 / 0.08 | 44.73 / 3.87 | 31.81 / 2.43 |
| C3 random, seed 44 | 64 | 72.90 / 1.28 | 65.99 / 0.09 | 47.95 / 3.93 | 35.60 / 2.72 |
| C4 C0-update basis | 4 | 74.86 / 49.28 | 74.38 / 46.19 | 70.79 / 29.84 | 69.46 / 30.39 |
| C4 C0-update basis | 8 | 74.94 / 49.56 | 74.87 / 46.89 | 70.98 / 29.66 | 70.06 / 29.94 |
| C4 C0-update basis | 16 | 74.97 / 49.17 | 74.82 / 45.74 | 70.96 / 29.16 | 70.48 / 28.91 |
| C4 C0-update basis | 32 | 74.93 / 49.00 | 74.97 / 45.20 | 70.91 / 29.66 | 70.14 / 29.68 |
| C4 C0-update basis | 64 | 74.75 / 48.92 | 75.03 / 45.04 | 71.04 / 29.72 | 69.94 / 30.28 |
| L0 retained-client basis | 4 | 72.59 / 1.16 | 64.64 / 0.07 | 44.95 / 2.99 | 27.94 / 1.56 |
| L0 retained-client basis | 8 | 72.76 / 1.23 | 65.82 / 0.07 | 51.09 / 2.14 | 29.34 / 0.89 |
| L0 retained-client basis | 16 | 73.15 / 1.36 | 66.71 / 0.10 | 52.49 / 2.00 | 29.85 / 0.63 |
| L0 retained-client basis | 32 | 73.54 / 1.66 | 67.63 / 0.24 | 56.48 / 1.78 | 34.30 / 0.69 |
| L0 retained-client basis | 64 | 73.72 / 2.02 | 68.98 / 0.43 | 59.02 / 2.26 | 39.21 / 0.76 |

### L1 and L2 curvature sweeps

GGN FLOPs below are in trillions and come from the cost record for the row. L2 costs are incremental stage costs. Add the default L1 spectrum cost of 14.70 trillion for a complete default-input L2 pipeline. The separate IID rank-20/512 GGN control and IID sweep measurement have the same displayed endpoints with a small difference in clean loss. The non-IID rank-20/512 comparison uses the sweep entry.

| Variant | IID | IID GGN TFLOPs | Non-IID | Non-IID GGN TFLOPs |
| --- | --- | --- | --- | --- |
| L1 GGN, rank 20, 512 samples | 68.84 / 1.33 | 1.47 | 41.09 / 12.48 | 1.47 |
| L1 GGN, rank 20, 2,048 samples | 69.39 / 1.21 | 5.88 | 37.22 / 12.37 | 5.88 |
| L1 GGN, rank 50, 512 samples | 69.68 / 1.29 | 3.67 | 43.12 / 5.86 | 3.67 |
| L1 default GGN, rank 50, 2,048 samples, seed 42 | 70.27 / 1.34 | 14.70 | 43.07 / 5.84 | 14.70 |
| L1 GGN, rank 100, 512 samples | 70.59 / 1.66 | 7.35 | 49.88 / 5.47 | 7.35 |
| L1 GGN, rank 100, 2,048 samples | 71.52 / 1.76 | 29.40 | 51.71 / 7.17 | 29.40 |
| L1 default repeat, seed 43 | 70.22 / 1.30 | — | 44.63 / 8.58 | — |
| L1 default repeat, seed 44 | 69.82 / 1.14 | — | 47.71 / 7.82 | — |
| L1 eval-BN GGN matched curvature control, rank 20, 512 | 68.84 / 1.33 | — | — | — |
| L1 exact Hessian, rank 20, 512 | 68.74 / 1.32 | — | 37.65 / 18.12 | — |
| L1 train-stat BN GGN, rank 20, 512 | 68.01 / 0.72 | — | 46.08 / 4.76 | — |
| L2 Taylor, W = 1, 512 examples | 10.04 / 97.77 | 0.04 | 10.00 / 0.00 | 0.04 |
| L2 Taylor, W = 2, 512 examples | 10.00 / 0.00 | 0.07 | 10.00 / 0.00 | 0.07 |
| L2 Taylor, W = 5, 512 examples | 10.00 / 100.00 | 0.18 | 10.00 / 100.00 | 0.18 |
| L2 Taylor, W = 10, 512 examples | 10.00 / 100.00 | 0.37 | 10.00 / 100.00 | 0.37 |
| L2 Taylor, W = 1, 2,048 examples | 10.17 / 19.42 | 0.15 | 10.40 / 0.00 | 0.15 |
| L2 Taylor, W = 2, 2,048 examples | 10.00 / 0.00 | 0.29 | 10.00 / 0.00 | 0.29 |
| L2 Taylor, W = 5, 2,048 examples | 10.00 / 100.00 | 0.73 | 10.00 / 100.00 | 0.73 |
| L2 Taylor, W = 10, 2,048 examples | 10.00 / 100.00 | 1.47 | 10.00 / 100.00 | 1.47 |
| L2 momentum transfer, W = 1, 512 examples | 73.00 / 1.83 | 5.81 | 63.74 / 3.84 | 6.87 |
| L2 momentum transfer, W = 2, 512 examples | 73.50 / 1.96 | 11.61 | 65.89 / 3.57 | 13.74 |
| L2 momentum transfer, W = 5, 512 examples | 73.79 / 2.00 | 29.03 | 67.77 / 3.47 | 34.36 |
| L2 momentum transfer, W = 10, 512 examples | 74.00 / 2.07 | 58.06 | 68.34 / 3.10 | 68.72 |
| L2 momentum transfer, W = 5, 2,048 examples | 74.42 / 2.17 | 116.13 | 69.83 / 2.88 | 137.44 |

### L3 continuation

| L0 input | IID | Non-IID | IID client TFLOPs | Non-IID client TFLOPs | IID wall seconds | Non-IID wall seconds |
| --- | --- | --- | --- | --- | --- | --- |
| default | 74.67 / 1.57 | 69.16 / 2.24 | 9.45 | 9.75 | 92.27 | 96.31 |
| half_scale | 74.93 / 3.19 | 69.67 / 3.74 | 9.45 | 9.75 | 93.19 | 96.30 |

### IID onset gates

The shift-20 group is post-hoc and evaluation-informed. Early multiplier 1 reproduces the ungated correction.

| Gate shift (rounds) | Early multiplier | Margin 0 | Margin 2 | Margin 5 |
| --- | --- | --- | --- | --- |
| 0 | 0 | 69.82 / 0.44 | 69.14 / 0.38 | 68.72 / 0.32 |
| 0 | 0.5 | 68.74 / 0.31 | 68.59 / 0.28 | 68.13 / 0.26 |
| 0 | 1 | 67.63 / 0.24 | 67.63 / 0.24 | 67.63 / 0.24 |
| 10 | 0 | 71.04 / 0.46 | 70.83 / 0.43 | 70.28 / 0.42 |
| 10 | 0.5 | 69.74 / 0.38 | 69.52 / 0.33 | 69.24 / 0.37 |
| 10 | 1 | 67.63 / 0.24 | 67.63 / 0.24 | 67.63 / 0.24 |
| 20 | 0 | 72.96 / 1.16 | 72.47 / 0.81 | 71.77 / 0.54 |
| 20 | 0.5 | 71.08 / 0.63 | 70.88 / 0.52 | 70.38 / 0.40 |
| 20 | 1 | 67.63 / 0.24 | 67.63 / 0.24 | 67.63 / 0.24 |

## Onset-free diagnostics

### Detector thresholds and outcomes

| Partition | κ | Threshold | Empirical benign alarm rate | Detected round | Error (rounds) |
| --- | --- | --- | --- | --- | --- |
| iid | 0.5 | 20.268 | 0.00 | 15 | -15 |
| iid | 1.0 | 8.214 | 0.00 | 12 | -18 |
| noniid | 0.5 | 44.751 | 0.00 | No alarm | — |
| noniid | 1.0 | 21.434 | 0.00 | No alarm | — |

### Benign validation and curvature quadratic

| Partition | Endpoint | Sample CA | Loss | Loss change | Half-Hessian quadratic |
| --- | --- | --- | --- | --- | --- |
| iid | Attacked input | 76.95 | 0.6827 | — | — |
| iid | c1 | 65.62 | 0.9560 | 0.2733 | 0.3986 |
| iid | l0 | 70.12 | 0.8397 | 0.1570 | 0.2785 |
| iid | l1 | 73.05 | 0.7736 | 0.0909 | 0.1567 |
| noniid | Attacked input | 71.09 | 0.7848 | — | — |
| noniid | c1 | 28.71 | 7.5151 | 6.7303 | 7.6166 |
| noniid | l0 | 37.11 | 2.8417 | 2.0569 | 1.7314 |
| noniid | l1 | 44.73 | 1.7568 | 0.9720 | 0.4680 |

### L1 per-layer cosine to counterfactual displacement

| Parameter tensor | IID | Non-IID |
| --- | --- | --- |
| features.0.weight | 0.1518 | 0.1707 |
| features.0.bias | 0.1346 | -0.0072 |
| features.1.weight | -0.0097 | -0.1264 |
| features.1.bias | 0.3039 | 0.2227 |
| features.4.weight | 0.1021 | 0.0648 |
| features.4.bias | 0.1472 | 0.1880 |
| features.5.weight | 0.0314 | -0.0676 |
| features.5.bias | 0.0478 | 0.1060 |
| classifier.0.weight | 0.0758 | -0.0180 |
| classifier.0.bias | -0.1359 | 0.0769 |
| classifier.3.weight | 0.1068 | 0.0310 |
| classifier.3.bias | 0.0819 | 0.3689 |
| Whole parameter vector | 0.0834 | 0.0125 |

Whole-vector C1 and L0 controls and the count of layer wins are given in the [measured report](https://github.com/anshul45-github/Federated-Backdoor-Unlearning/blob/f81d1fdfa3d10630d4438cd8df6a600dc51e3c4c/results/onsetfree/REPORT.md). The compact JSON stores the L1 per-layer series. The original diagnostic figures include the controls.

### ASR by non-target source class

#### IID

| Model | Overall | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attack | 51.93 | 48.2 | 61.0 | 56.1 | 54.9 | 46.5 | 35.4 | 46.2 | 76.5 | 42.6 |
| counterfactual | 3.08 | 2.5 | 6.5 | 1.9 | 2.0 | 1.1 | 0.6 | 1.4 | 7.4 | 4.3 |
| federaser_ecal1 | 2.33 | 1.9 | 5.2 | 1.2 | 1.5 | 0.7 | 0.3 | 0.9 | 6.3 | 3.0 |
| federaser_ecal2 | 2.80 | 1.6 | 6.2 | 2.1 | 1.8 | 1.1 | 0.4 | 1.0 | 7.9 | 3.1 |
| c1 | 0.11 | 0.0 | 0.5 | 0.1 | 0.1 | 0.0 | 0.0 | 0.0 | 0.3 | 0.0 |
| l0_default | 0.24 | 0.0 | 0.8 | 0.1 | 0.2 | 0.0 | 0.0 | 0.1 | 0.7 | 0.3 |
| l0_half_scale | 1.66 | 0.6 | 3.2 | 0.9 | 1.5 | 0.8 | 0.2 | 1.4 | 4.1 | 2.2 |
| l1_default | 1.34 | 0.7 | 2.5 | 1.0 | 1.0 | 0.5 | 0.2 | 0.7 | 4.1 | 1.4 |
| l3_default | 1.57 | 0.5 | 3.8 | 1.1 | 0.9 | 0.5 | 0.2 | 0.5 | 4.8 | 1.8 |

#### Non-IID

| Model | Overall | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attack | 29.32 | 29.7 | 35.0 | 31.8 | 33.2 | 18.7 | 10.0 | 37.5 | 40.5 | 27.5 |
| counterfactual | 3.41 | 3.0 | 7.5 | 3.0 | 3.6 | 1.7 | 0.8 | 2.1 | 6.0 | 3.0 |
| federaser_ecal1 | 2.51 | 2.0 | 6.0 | 1.6 | 3.3 | 1.1 | 0.3 | 1.5 | 4.4 | 2.4 |
| federaser_ecal2 | 2.59 | 2.8 | 5.3 | 2.2 | 2.5 | 1.1 | 0.4 | 1.4 | 4.0 | 3.6 |
| c1 | 1.06 | 2.1 | 1.7 | 1.1 | 1.0 | 0.2 | 0.1 | 0.6 | 0.2 | 2.5 |
| l0_default | 0.69 | 1.3 | 1.8 | 0.1 | 0.6 | 0.0 | 0.1 | 0.3 | 0.3 | 1.7 |
| l0_half_scale | 1.78 | 2.3 | 3.6 | 1.0 | 2.4 | 0.3 | 0.1 | 1.8 | 1.3 | 3.2 |
| l1_default | 5.84 | 12.3 | 8.4 | 1.4 | 3.6 | 0.8 | 0.9 | 7.0 | 6.6 | 11.6 |
| l3_default | 2.24 | 2.5 | 4.5 | 1.6 | 2.7 | 0.8 | 0.2 | 1.0 | 4.2 | 2.7 |

Each non-target source class has 1,000 images. Figure pairs for energy, spectrum, cosine, benign validation, Taylor and stable windows, cost and onset threshold trade-offs are preserved in the [onset-free plots](https://github.com/anshul45-github/Federated-Backdoor-Unlearning/tree/f81d1fdfa3d10630d4438cd8df6a600dc51e3c4c/results/onsetfree/plots).
