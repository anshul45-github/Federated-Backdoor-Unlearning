# Stage 1 code and artifact audit

Status: code audit and matched trajectory regeneration complete; matched FedEraser baselines pending. Sources are `configs/v2_{iid,noniid}.yaml`, `scripts/run_experiment_v2.py`, `scripts/baseline/run_federaser_v5.py`, `src/training/{fedavg,train_utils}.py`, `src/models/small_cnn.py`, and saved artifact metadata.

| Item | Observed implementation |
|---|---|
| Local training | SGD, learning rate 0.01, momentum 0.9, implicit weight decay 0, batch size 64, two local epochs. Optimizer state is recreated for every client and round. |
| Steps per client-round | `2 * ceil(n_i / 64)` for nonempty clients: 158 IID; for saved non-IID checkerboard partition C0-C9: 114, 180, 90, 90, 204, 128, 248, 128, 116, 274. V2 non-IID partition counts still need direct verification. |
| FL | 50 rounds; all 10 clients participate. FedAvg uses `p_i = n_i / sum_j n_j`; IID `p_i=0.1`. Saved non-IID C0 count is 3,597/50,000, hence `p_m=0.07194`. FedEraser excludes C0 and recomputes weights from C1-C9 counts. |
| Evaluation-only attack metadata | V2 config activates attack at round 30, poisons 10% of C0 examples, uses a 3x3 white patch, and targets CIFAR-10 class 0 (airplane). Never pass these to unlearning selection. |
| FedEraser | Retains rounds 1-50. `E_cal` is 1 or 2; round 1 reuses the stored update, subsequent rounds calibrate a new C1-C9 local update. `calibrate_update` rescales all trainable float tensors using one whole-vector L2 norm, excluding running-stat buffers and counters. |
| BatchNorm | `compute_model_delta` stores every state-dict key, including running means, running variances and integer counters converted to float. Main FedAvg aggregates float deltas and rounds integer counters on casting. FedEraser calibration does not scale running-stat deltas and its aggregation preserves integer counters. Parameter-only onset-free methods must state their BN policy explicitly. |
| Stored trajectory | `trajectory/round_001` to `round_050`, each with `delta_client_00.pt` through `delta_client_09.pt` and `global_model.pt` (post-round state). `initial_model.pt` is pre-round 1. Complete checkerboard directories contain 550 files each, including 500 deltas. The V2 `phase2/*_attack_logged` directories have no trajectory subdirectory in this checkout. |
| Model | SmallCNN: Conv(3,32,3), BN32, pool, Conv(32,64,3), BN64, pool, Linear(4096,128), dropout 0.5, Linear(128,10). Trainable parameter count from architecture: 545,290 (verify programmatically when PyTorch is available). |
| Metrics | Clean accuracy is top-1 accuracy over all 10,000 CIFAR-10 test images. ASR is the target prediction rate over 9,000 triggered, non-target test images. Both use test data only for reporting. |

The original V2 saved endpoints record attacked IID 75.01% CA / 56.0667% ASR and non-IID 69.8% / 32.2111%. These come from `results/phase2/*_attack_logged/final_results.json`; they are **not** results of regenerated trajectories. The checkerboard trajectories are from a different trigger and cannot be used for a matched V2 claim.

Local setup at audit time: Python 3.9.13, no usable PyTorch/GPU runtime, and about 0.56 GB free on E:. The existing two checkerboard trajectories occupy 1,203,304,700 bytes each. This prevented local regeneration; the remote run below resolved it.

## Designated remote-folder execution

The user provided SSH access restricted to `/DATA/teaching/new-approach`. A clean clone and all new data, dependencies, logs and results were placed within that folder. The remote runtime is Python 3.10.12, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124 and NVIDIA RTX A5000. This differs from the PDF's suggested Python/PyTorch versions; the clone's `requirements.txt` does not pin those versions. At setup the folder's filesystem had 403 GB available.

Regenerated white-patch attacked trajectories are at `results/onsetfree/phase2_iid_attack` and `results/onsetfree/phase2_noniid_attack` in that remote clone. `verify_trajectory.py` recorded 50 rounds, 500 client deltas, zero maximum float/integer reconstruction error, and zero final-endpoint mismatch for each. Their attack endpoints were IID **74.70% CA / 51.9333% ASR** and non-IID **71.59% / 29.3222%**, from their `final_results.json`. Relative to the older saved V2 references, IID differs by -0.31 CA and -4.13 ASR points; non-IID differs by +1.79 CA and -2.89 ASR points. The gap is reported rather than assumed away.

The local environment paragraph above is a historical setup blocker; the remote environment resolved execution. Matched counterfactual and FedEraser results are being produced separately before `stage1.DONE` is set.
