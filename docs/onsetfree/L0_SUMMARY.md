# L0 interim result: ASR removal with substantial clean-accuracy loss

These are new matched white-patch trajectories on the designated remote GPU host, not the older endpoint references. CA and ASR are test percentages, used here for reporting only. The L0 method was generated from client ID, stored updates and benign deltas, with default rank 32, scale 1, and attacked BatchNorm buffers kept.

| Endpoint | IID CA | IID ASR | Non-IID CA | Non-IID ASR |
|---|---:|---:|---:|---:|
| Attacked | 74.70 | 51.93 | 71.59 | 29.32 |
| C1: parameter-only subtraction | 64.31 | 0.11 | 27.15 | 1.06 |
| C2: contrast only | 64.46 | 0.08 | 26.69 | 1.06 |
| C3: random subspace, seed 42 | 65.37 | 0.09 | 29.58 | 2.23 |
| C4: malicious-update subspace | 74.97 | 45.20 | 70.13 | 29.68 |
| L0 default, k=32, s=1 | 67.63 | 0.24 | 34.30 | 0.69 |
| L0 + benign BN recalibration | 67.53 | 0.41 | 43.23 | 0.74 |

Default L0 gains **3.32** IID and **7.15** non-IID clean-accuracy points over the parameter-only C1 control, while C1 has lower IID ASR and higher non-IID ASR. It therefore does **not** dominate C1. L0 has higher CA than the three rank-32 random-subspace controls in both partitions, but their IID ASR is lower. C4 retains CA largely because it leaves the backdoor: its ASR remains 45.2% IID and 29.68% non-IID. BN recalibration helps non-IID CA by 8.93 points over default L0 but still leaves a large gap to the attacked model.

The pre-registered rank/scale sweep shows a strong scale effect: k=32, s=0.5 yields **73.54% CA / 1.66% ASR** IID and **56.48% / 1.78%** non-IID. At k=64, s=0.5 the figures are **73.72% / 2.02%** IID and **59.02% / 2.26%** non-IID. These are registered ablations, not test-selected variants. No hyperparameter was chosen by ASR. The full 54-checkpoint-per-partition sweep and three random seeds are recorded in `results/onsetfree/l0_sweep/{iid,noniid}/evaluation.json` on the remote host.

Matched FedEraser Ecal 1 reached 71.99% CA / 2.33% ASR IID and 67.69% / 2.51% non-IID. Thus default L0 meets the prespecified S0 CA/ASR threshold for IID only. The registered half-scale IID L0 reaches 73.54%/1.66% with zero additional client-training FLOPs, meeting S1 on this single trajectory; the non-IID half-scale result does not. This is a registered ablation, not a test-selected default.

On the fixed 512-example benign sample, C1 versus L0 actual loss increase was **0.2733 versus 0.1570** IID and **6.7303 versus 2.0569** non-IID. Their Hessian quadratics were **0.3986 versus 0.2785** IID and **7.6166 versus 1.7314** non-IID. The sample is disjoint from the L1 curvature sample but comes from the original training clients, so it is not a genuinely held-out population. D1 per-layer energy is in each L0 metadata JSON; D2 per-layer cosine and D3 validation/Hessian JSON exist for both partitions, with PNGs under remote `results/onsetfree/plots/`. D4 CA/ASR, server wall time and zero client FLOPs are recorded. Each matched trajectory occupies 1,203,619,196 bytes; each default L0 output directory is about 13.16 MB and each full sweep about 118.17 MB.

**Plain verdict:** default L0 substantially reduces ASR but does not dominate C1, FedEraser or the counterfactual. It sacrifices too much non-IID CA. A registered half-scale variant is promising on IID only; this does not establish reproducibility across attack seeds. The C1 control intentionally keeps attacked BN buffers, whereas the repository's historical subtraction baseline changes some floating BN buffers. Evidence is in remote `results/onsetfree/{l0,l0_sweep,diagnostics,plots}/` and the local matched `baselines.json`.
