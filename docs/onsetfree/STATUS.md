# Execution status, 2026-11-17 (India time)

| Stage | State | Evidence / next required gate |
|---|---|---|
| 1 | Partial | Remote IID/non-IID V2 trajectories regenerated and verified at 50 rounds / 500 deltas each with zero reconstruction error. IID counterfactual completed; non-IID counterfactual and four matched FedEraser settings are still running/pending. PyTorch tests pass remotely. No `stage1.DONE` yet. |
| 2 | Complete at specified abstract level | 11 requested and four additional primary abstracts reviewed in `NOVELTY_CHECK.md`; `stage2.DONE` records its limited scope. |
| 3 | Partial | Default L0/C1-C4, rank/scale sweeps, three random seeds and BN recalibration evaluated for both partitions; see `L0_SUMMARY.md`. D1 and IID D2 saved. Full benign validation/Hessian D3, non-IID D2 and plots remain; no `stage3.DONE`. |
| 4 | Partial | GGN/Hessian products and spectral/recursion unit tests pass remotely. Default L1 and L2 W/Bc sweeps are queued after matched baselines. Full curvature and BN ablations remain. |
| 5 | Partial, optional | Leave-one-out CUSUM run for both partitions. IID alarmed 15/18 rounds early; non-IID did not alarm. See `ONSET_SUMMARY.md`; gating ablation remains. |
| 6 | Pending | No S0/S1/S2 or P1-P4 verdict can be made without L0-L2 measurements. |

The remote clone is contained under `/DATA/teaching/new-approach`. Its white-patch V2 trajectories are new, matched sources; the existing checkerboard trajectories in the local checkout are different 4x4-trigger runs. Pre-existing uncommitted local changes were left untouched.
