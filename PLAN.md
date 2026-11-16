# Onset-free federated backdoor unlearning

Branch: `onsetfree`. Source proposal: `Onset_Free_Federated_Unlearning.pdf` and the six-stage pasted specification. The PDF is a research proposal, not an established result. Its remote-SSH and `nohup` assumptions do not describe this Windows checkout.

## Invariants

- The unlearning algorithm receives the malicious client ID, stored states and updates, and benign client data only. Attack onset, trigger, target label, test CA and test ASR are evaluation-only.
- Use the registered defaults before any sweep. Select variants using a held-out benign validation split; label any ASR-selected variant `oracle-tuned`.
- Keep outputs in `results/onsetfree/`, documentation in `docs/onsetfree/`, and new runners in `scripts/onsetfree/`. Preserve all existing results and unrelated worktree changes. A `.DONE` marker means all stage deliverables were actually verified.
- Record source paths, config, seed, checkpoint hashes, timing and compute method with each result. Do not mix the 3x3 white-patch V2 and 4x4 checkerboard trajectories.

## Environment and data gate

2026-11-16 audit: this Windows checkout has Python 3.9.13 but no PyTorch in the available Python installations, no usable GPU, and only about 0.56 GB free on E:. Two complete checkerboard trajectories already occupy about 2.4 GB, but the V2 white-patch `phase2/*_attack_logged` directories have no trajectories. The proposal's claimed Python 3.11.6 / PyTorch 2.14.0 pin does not match this checkout's `requirements.txt`; existing GPU logs report PyTorch 2.8.0+cu128. Before any long experiment, use a host with a compatible PyTorch/GPU and enough free space for new trajectories, checkpoints and results. The user will provide SSH access with one designated folder; all remote file access and writes must stay within that folder. Do not mark a stage done based on old endpoint summaries.

## Stage gates

1. **Audit and matched baselines.** Finish `docs/onsetfree/AUDIT.md`, trajectory loader and reconstruction test. Regenerate V2 IID and non-IID 3x3 trajectories (500 deltas each), then run matched FedEraser Ecal 1/2, subtraction and counterfactual, recording CA, ASR, wall clock and client FLOPs in `results/onsetfree/baselines.json`. Verify trajectory reconstruction and source hashes. Mark `stage1.DONE` only then. Existing checkpoint metrics are reference values, not matched new baselines.
2. **Literature check.** Open the specified primary abstracts plus 3-5 directly relevant papers, document verified overlap and onset assumptions in `docs/onsetfree/NOVELTY_CHECK.md`, then mark `stage2.DONE`. Treat the PDF's literature notes as leads, not citations.
3. **L0.** Implement uncentered per-layer benign-update SVD projection and C1-C4, with default `k=32, s=1`, both BN variants, registered rank/scale sweeps, three random seeds and D1-D4 diagnostics. Save JSON, PNG and a plain verdict in `docs/onsetfree/L0_SUMMARY.md`, then mark `stage3.DONE`.
4. **L1/L2.** Implement and test GGN/Hessian-vector products, eigenpairs, momentum-aware contraction and time-varying recursion. Run registered `k_L`, `B_c`, `W`, curvature and BN ablations, report P1/P2/P4 and compute in `docs/onsetfree/L1_L2_SUMMARY.md`, then mark `stage4.DONE`. Derive and validate the effective local step under momentum rather than silently using the no-momentum formula.
5. **Optional onset ablation.** Fit a leave-one-out benign null, calibrate CUSUM false alarm without attack metadata, test gates and the deliberately late onset. Report P3 in `docs/onsetfree/ONSET_SUMMARY.md`, then mark `stage5.DONE` if completed.
6. **Final evaluation.** Compare all methods on the same trajectory and evaluation populations, use at least three seeds for stochastic components, calculate S0/S1/S2 and P1-P4, account for client FLOPs, server work, storage and communication. Publish `results/onsetfree/REPORT.md` with exact commands and limitations; mark `stage6.DONE` only after verification.

## Current progress

- Branch created without discarding pre-existing worktree changes.
- The designated remote folder now contains a clean `onsetfree` checkout, CIFAR-10 data and isolated dependencies. All new remote project files remain under `/DATA/teaching/new-approach`. Both V2 white-patch trajectories passed 50-round / 500-delta reconstruction checks; their endpoints and L0 variants have been evaluated.
- Stages 1-6 are complete as experiments and reporting, including negative L0/L1/L2 and onset findings. Both matched white-patch trajectories, counterfactuals and all four FedEraser settings are verified in `baselines.json`. Registered L0, L1, L2 and L3 runs, three stochastic L1 seeds, controls, per-class ASR, benign validation, compute/storage accounting and 16 plots are measured in the remote folder and compacted in `summary_metrics.json`. The report states S0-S2/P1-P4 outcomes and limitations; 14 focused tests passed. Only genuine completed gates receive `.DONE` markers.
