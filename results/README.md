# Experiment runs

| Experiment | Config | Saved outputs |
|---|---|---|
| Phase 1 IID and non-IID baselines | [IID](../configs/v2_iid.yaml), [non-IID](../configs/v2_noniid.yaml) | [Baseline runs and figures](phase1/), [summary](phase1/phase1_summary.csv) |
| Phase 2 unlearning and retraining | [Baseline protocol](../docs/baseline/EXPERIMENT_PROTOCOL.md) | [IID table](phase2/phase2_iid.csv), [non-IID table](phase2/phase2_noniid.csv), [FedEraser registry](phase2/federaser/RUN_REGISTRY.md), [corrected evaluation](phase2/phase2_evaluation_corrected.json), and [superseded evaluation](phase2/phase2_evaluation_superseded.json) |
| Smoke test | [Smoke config](../configs/v2_smoke.yaml) | [Smoke outputs](smoke/) |
| 4x4 checkerboard IID and non-IID | [IID](../configs/v3_checkerboard4_iid.yaml), [non-IID](../configs/v3_checkerboard4_noniid.yaml) | [Timestamped runs](checkerboard/). The IID rerun and non-IID run have local trajectories. |
| Extreme label skew | [Stress config](../configs/v4_checkerboard4_extreme_noniid.yaml) | [Timestamped runs](label_skew/). The completed seed-42 run has a local attack trajectory. |

The [runtime ledger](runtime_ledger.csv) covers multiple phases. Recorded output files retain their original metadata, which can include paths from the original checkout. The [historical checksum manifest](../manifests/historical/sha256_manifest.txt) also retains its original paths.

## Original baseline snapshot notes

These CSVs transcribe the current project result record available for this handoff. They are intended for team reference and plotting. They are not regenerated from experiment source in this initial documentation commit because the original training scripts, saved run folders and raw logs have not yet been imported here.

Before using them as final published numbers, cross-check each row against the canonical files from the experiment working tree and update the artifact manifest. Preserve the current copies as a reference snapshot; if any value changes after verification, add a dated correction note rather than silently overwriting history.

## Files

- phase1_summary.csv — clean vs attacked training.
- phase2_iid.csv — six IID endpoints and their main metrics.
- phase2_noniid.csv — six Non-IID endpoints from one saved partition / seed.
- federaser_corrected_runs.csv — corrected FedEraser Ecal=1/2 run summaries.
- runtime_ledger.csv — timer values and the work each timer includes.

## Metric denominators

- Clean accuracy: all 10,000 clean CIFAR-10 test images.
- ASR/TTR: 9,000 triggered non-Airplane images predicted as Airplane (class 0).
- UTR: 9,000 untriggered non-Airplane images predicted as Airplane (class 0).
- Airplane recall: 1,000 true Airplane images.
- Counterfactual agreement: argmax prediction agreement with M_CF on the named population; not accuracy or proof of erasure.

The two 9,000-image non-Airplane populations are alternate views (with and without trigger), not separate disjoint examples from each other.
