# Execution status, 2026-11-17 (India time)

| Stage | State | Evidence / next required gate |
|---|---|---|
| 1 | Partial | Both V2 trajectories, both counterfactuals, and FedEraser Ecal 1 for IID/non-IID and Ecal 2 for IID completed. Non-IID Ecal 2 is running. Both trajectories passed 50-round / 500-delta exact reconstruction checks. No `stage1.DONE` yet. |
| 2 | Complete at specified abstract level | 11 requested and four additional primary abstracts reviewed in `NOVELTY_CHECK.md`; `stage2.DONE` records its limited scope. |
| 3 | Partial | Default L0/C1-C4, rank/scale sweeps, three random seeds and BN recalibration evaluated for both partitions; see `L0_SUMMARY.md`. D1/D2 for both partitions and IID D3 saved. Non-IID D3 and plots remain; no `stage3.DONE`. |
| 4 | Partial | IID default L1 reached 70.27% CA / 1.34% ASR; IID L2 first-order W/Bc sweep collapsed to roughly 10% CA. GGN/Hessian and recursion tests pass. Non-IID and full ablations remain; see `L1_L2_SUMMARY.md`. |
| 5 | Complete negative result | Leave-one-out CUSUM run for both partitions. IID alarmed 15/18 rounds early; non-IID did not alarm. Twenty-seven IID gates, including the registered +10 shift and post-hoc +20 shift, were evaluated; threshold trade-off plots were rendered. P3 refuted. See `ONSET_SUMMARY.md` and `stage5.DONE`. |
| 6 | Partial | Registered L0 variants and three-round benign L3 repair have measured results, but a complete final table and remaining ablations are pending. |

The remote clone is contained under `/DATA/teaching/new-approach`. Its white-patch V2 trajectories are new, matched sources; the existing checkerboard trajectories in the local checkout are different 4x4-trigger runs. Pre-existing uncommitted local changes were left untouched.
