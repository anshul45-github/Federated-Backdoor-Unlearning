# Conv2 + BatchNorm historical component patching

This experiment repairs the saved checkerboard attack model by blending historical Conv2/BN components into its final state. It never edits the source checkpoints or earlier result files. Both existing runs have a final **G50**, with G10, G20 and G29 available as donors. All three donors precede attack activation at round 30.

## Reported starting results

The following full-test results were supplied by the researcher. They are context, not outputs of the new split-based patch search.

| Model | IID CA % | IID ASR % | Non-IID CA % | Non-IID ASR % |
|---|---:|---:|---:|---:|
| Attacked | 75.26 | 92.18 | 71.14 | 80.89 |
| FedEraser | 75.19 | 1.18 | 70.18 | 0.94 |
| KD | 73.09 | 0.41 | 59.35 | 0.79 |
| Subtraction | 68.21 | 0.02 | 31.25 | 0.13 |

## Run both stages

Run from the repository root in the same Python environment used for attack training. Dependencies are already listed in `requirements.txt`; the local extracted CIFAR-10 dataset is required. No attack retraining is needed.

```powershell
$IID = "results/checkerboard/iid_20261024_015516_rerun_001"
$NonIID = "results/checkerboard/noniid_20261024_150813_189"

# Optional input/configuration/checkpoint checks; no model evaluation or output folder.
python scripts/run_component_patching.py --source_run "$IID" --check_only
python scripts/run_component_patching.py --source_run "$NonIID" --check_only

# Stage 1 grid, then Stage 2 for the best Stage 1 donor/alpha.
python scripts/run_component_patching.py --source_run "$IID" --ca_budget_pp 1.0
python scripts/run_component_patching.py --source_run "$NonIID" --ca_budget_pp 1.0
```

Defaults: Stage 1 donors G10/G20/G29; alpha 0.25/0.5/0.75/1.0; Stage 2 selects one donor/alpha; 1.0 percentage-point CA-loss budget; 50/50 class-stratified selection/report split; ten random orders; no calibration. Device selection is automatic; use `--device cuda` or `--device cpu` to override. `--batch_size 128` reduces evaluation memory. `--data_dir PATH` changes the dataset root.

Output defaults to `<source_run>/component_patching`. Repeated invocations get fresh numbered sibling directories, such as `component_patching_rerun_001`; the chosen absolute path is printed. You can supply `--output_dir PATH`. The run does not resume partial searches.

## Stage 1: whole Conv2 + BN block

For each of the twelve donor/alpha combinations, the runner constructs a fresh candidate from the final attacked model:

`candidate[channel] = (1 - alpha) * final[channel] + alpha * donor[channel]`

Stage 1 applies this to all 64 output channels. Alpha 1 is exact replacement. Conv2 is `features.4`, and its associated BN is `features.5` in `SmallCNN`. The six interpolated tensors are:

| State key | Patched component |
|---|---|
| `features.4.weight` | Conv2 output filter, including every input-channel kernel |
| `features.4.bias` | Conv2 output bias |
| `features.5.weight` | Matching BN scale |
| `features.5.bias` | Matching BN shift |
| `features.5.running_mean` | Matching BN running mean |
| `features.5.running_var` | Matching BN running variance |

The scalar BN `num_batches_tracked` stays from the final model. Conv1, its BN, and both classifier layers remain unchanged. Output-channel identities are inherited from the same training trajectory; no channel matching/permutation search is performed. BN statistics are blended directly, with no BN recalibration by default.

CA and ASR are printed and written to `stage1.csv`, along with signed `delta_ca_pp` and `delta_asr_pp` relative to the original model on the **selection split**. Negative delta ASR means improvement. All metrics are percentages, and deltas are percentage points.

```powershell
# Stop after the twelve Stage 1 candidates; never evaluate the report split.
python scripts/run_component_patching.py --source_run "$IID" --stage 1
```

## Stage 2: individual channels and prefix curves

By default, Stage 1 chooses one donor/alpha using the largest regularized ASR-drop/CA-loss score. Ties prefer lower ASR, higher CA, a later donor, then smaller alpha. A whole-block candidate can exceed the CA budget and still be chosen for channel exploration, because smaller subsets may fit the budget.

For the chosen donor/alpha, every channel 0 through 63 is patched independently into the original final model. Its score is:

`score = (baseline_ASR - candidate_ASR) / max(max(0, baseline_CA - candidate_CA), epsilon)`

Default epsilon is `1e-6` percentage points, configurable with `--score_epsilon_pp`. Zero-loss and CA-improving channels therefore receive very large scores when they reduce ASR. This is a ranking heuristic; retain the raw CA/ASR deltas when interpreting it. Single-channel effects need not add up when channels are combined.

The runner evaluates every prefix length **k=0 through k=64** for:

1. Single-channel score, descending; ties use larger ASR drop, smaller CA loss, then channel ID.
2. Weight drift, descending: raw L2 distance between final and donor Conv2 output kernel plus bias. BN values are excluded from this distance to avoid mixing their scales with kernel weights.
3. Ten random channel orders, generated using seeds `314159 + shuffle_index`.

Every prefix starts from the final model; it is not applied on top of the previous candidate. Identical channel subsets share cached metrics. Each setting produces 64 single-channel rows and 780 prefix rows (12 orders times 65 lengths). This requires hundreds of evaluations, despite avoiding training. Searching all twelve donor/alpha settings costs substantially more.

Selection chooses the lowest selection ASR among **all tested prefix candidates, including random controls**, subject to selection CA loss <= the budget. Ties prefer higher CA, fewer channels, the score order, then drift, then random; later donors and smaller alpha break remaining ties. The unchanged k=0 model is always eligible, so an unsuccessful search returns an unchanged model. A random order can win; inspect `best_prefix_per_order.json` when comparing ranking methods. The ten random orders also have more search opportunities than either deterministic order, so a best-random result alone cannot establish ranking superiority.

```powershell
# Exhaustive donor/alpha exploration in Stage 2, with the same selection/report split.
python scripts/run_component_patching.py --source_run "$IID" --stage2_scope all --ca_budget_pp 1.0

# Or run Stage 2 alone for a preselected donor/alpha (example, not a recommendation).
python scripts/run_component_patching.py --source_run "$IID" --stage 2 --stage2_donor_round 29 --stage2_alpha 0.5 --ca_budget_pp 1.0
```

The same flags apply to `$NonIID`. Change the budget with `--ca_budget_pp 0.5` or `--ca_budget_pp 2.0`; 1.0 means a drop from selection CA 75% to 74% is allowed, not a 1% relative reduction. Full-test CA from the earlier experiment is not the budget baseline.

## Selection and held-out reporting

The CIFAR-10 test set is stratified by class using seed 2026. Default selection has 5,000 images (500/class), with 5,000 disjoint report images. Each ASR denominator excludes the original target class: 4,500 triggered images per default split. The trigger and target come from the source configuration, including the normalized 4x4 bottom-right checkerboard.

All Stage 1 screening, channel scoring, prefix choices, and optional calibration acceptance use selection images only. After the final patch and its metadata are saved, the original and selected models are evaluated once each on the report split. Report metrics never revise the chosen patch. A selection budget does not guarantee the same CA loss on the report split; both the loss and its budget check are saved.

This is a held-out split for this search. The original attack/baseline experiments already evaluated the complete test set, so this is not a previously untouched external test set. Repeatedly changing choices after inspecting report results would also compromise its held-out role. Keep the same split seed across IID and Non-IID comparisons, and compare baselines on identical splits rather than directly subtracting the earlier full-test percentages.

## Optional clean calibration

The first experiment defaults to patching only. To add a fixed number of clean calibration minibatches, append, for example:

```powershell
python scripts/run_component_patching.py --source_run "$IID" --ca_budget_pp 1.0 --calibration_steps 20 --calibration_lr 0.0001
```

Calibration occurs after selecting a prefix and uses clean CIFAR-10 **training** images from retained clients C1-C9, taken from the source partition file. It excludes C0 and uses no test images for gradients. Batch size is 64; SGD has zero momentum and weight decay. Only selected Conv2 output filters/biases and their BN affine parameters receive updates. All other parameters and unselected channels stay fixed; all BN running statistics stay fixed, and dropout is disabled. Assertions verify the unchanged tensors/slices.

The calibrated proposal is accepted only if it remains within the selection CA budget and improves selection ASR, or ties ASR with better CA. Otherwise the uncalibrated patch is retained. The report split is evaluated only after this decision. With k=0, calibration is skipped. Calibration is a separate experimental variant and is recorded in the channel metadata and manifest.

## Saved artifacts

| File | Contents |
|---|---|
| `manifest.json` | Arguments, source configuration, device/Torch version, source/donor paths and SHA-256 hashes, completion status |
| `split_indices.json` | Exact selection/report IDs, split seed, label and normalized-pixel hashes |
| `selection_baseline.json` | Original-model selection CA/ASR and denominators |
| `stage1.csv` | Twelve whole-block candidates with CA, ASR, deltas and score; absent for Stage 2-only runs |
| `stage2_settings.json` | Donor/alpha setting(s) actually searched |
| `single_channel_scores.csv` | Individual channel metrics, scores and weight drift |
| `ranking_orders.json` | All channel permutations for every searched setting |
| `channel_curves.csv` | Every prefix, its exact channels, metrics, score and CA-budget flag |
| `best_prefix_per_order.json` | Best feasible prefix for each ranking and donor/alpha |
| `best_patch_model.pt` | Selected full `SmallCNN` state dictionary; CPU tensors |
| `best_patch_channels.json` | Selected channels/donor/alpha/ranking, selection metrics, calibration details, hashes |
| `best_patch_uncalibrated.pt` | Pre-calibration state, only when calibration is attempted |
| `held_out_results.json` | Original and selected report metrics, signed deltas, report-budget flag |
| `channel_curves.png` | Selected setting's selection CA/ASR curves; random mean and +/-1 standard deviation |
| `channel_curves_G*_alpha*.png` | Separate plot for each searched donor/alpha setting |
| `report.md` | Human-readable selection/report results and selected patch |

Stage 1-only runs stop after the manifest, split/baseline and Stage 1 CSV. They do not create a best patch or report-split results. Stage 2/all-grid runs keep all setting curves in the same CSV. A manifest marked `started` indicates that the run did not finish; do not treat partial CSVs as completed experiments.

## Interpretation and validation

These historical donors still contain C0's clean pre-attack contributions. This experiment tests selective backdoor repair; it does not establish complete removal of the client's influence. A successful patch also does not prove that the patched channels alone encode the backdoor, because interpolation changes feature interactions and BN statistics.

Implementation: `scripts/run_component_patching.py`; reusable patch/selection helpers: `src/unlearning/component_patching.py`.

```powershell
python -m unittest discover -s tests -p test_component_patching.py -v
```

The suite checks stratified split separation, scoring and deterministic rankings, budget selection and k=0 fallback, complete mocked search output, and (when Torch is installed) exact patch boundaries, interpolation, counter preservation and evaluation index isolation. During implementation, both real source folders passed checkpoint/configuration preflight. Seven standard-library tests passed; four Torch tests were skipped because this agent's Python has no Torch. Numerical patching, optional calibration and plot generation therefore still require execution in the training environment. No new CA/ASR results are claimed here.
