# L1/L2 measured results on the new white-patch trajectories

The registered default L1 uses rank 50, 2,048 retained-client examples, eval-mode BatchNorm and a momentum-aware heavy-ball transfer. It ran without trigger, target, or attack-onset metadata. The IID endpoint reached **70.27% CA and 1.34% ASR**, versus 67.63%/0.24% for default L0, 64.31%/0.11% for C1 and 71.99%/2.33% for matched FedEraser Ecal 1. The L1 computation used 100 GGN-vector products over 32 batches, counted as 14,699,986,944,000 forward/backward-equivalent ATen FLOPs; matched IID FedEraser Ecal 1 used 77,183,608,320,000 client FLOPs. The ratio is about 5.25, below the prespecified 10x S1 saving.

The same non-IID default L1 reached **43.07% CA / 5.84% ASR**, far below matched FedEraser Ecal 1's 67.69% CA / 2.51% ASR and below its own attacked model's 71.59% CA. All 50 non-IID stiff modes had contraction below 0.5 (maximum 0.001457). This is a negative result for the default L1 outside IID.

On fixed 512-example benign samples disjoint from the L1 curvature samples, C1 increased cross-entropy by 0.2733 IID and 6.7303 non-IID, whereas L1 increased it by 0.0909 and 0.9720. The Hessian quadratics were 0.3986/7.6166 for C1 versus 0.1567/0.4680 for L1. This supports P1 on both measured samples, despite the poor non-IID test CA of L1. The samples are disjoint from curvature estimation but drawn from original training clients, so they are not genuinely held-out populations.

IID whole-vector cosine to the evaluation-only counterfactual displacement was 0.0834 for L1, versus 0.2353 for C1 and 0.1803 for L0. Non-IID whole-vector cosines were 0.0125, 0.1211 and 0.1498. In each partition, among the 10 layers with all three cosines defined, L1 beat both C1 and L0 in only 2. Two L0 bias-layer displacements were zero, so their cosines are undefined. **P2 is refuted on both measured trajectories**, globally and by a majority of comparable layers.

All 50 registered IID L1 eigenmodes had contraction below 0.5; the maximum recorded contraction was 0.0002802. Under this fixed-curvature heavy-ball approximation, stiff modes have effectively a one-round memory. The L2 Taylor collapse cannot validate a W saturation claim; the separate momentum-aware transfer is being swept across W.

In the matched rank-20, 512-example curvature ablation, exact Hessian versus GGN gave 68.74% CA / 1.32% ASR versus 68.84% / 1.33% IID, and 37.65% / 18.12% versus 41.09% / 12.48% non-IID. With train-mode BatchNorm batch statistics but dropout disabled, GGN gave 68.01% / 0.72% IID and 46.08% / 4.76% non-IID, versus the same eval-mode controls at 68.84% / 1.33% and 41.09% / 12.48%. The comparison is specific to rank 20 and 512 examples.

The registered L2 first-order Taylor window sweep gave approximately **10% CA** at every W in {1,2,5,10} and Bc in {512,2048} in both partitions; ASR varied between 0% and 100%. This is a failed, unstable approximation under the recorded `eta_eff=eta/(1-momentum)` and weighted local step count. It must not be treated as a successful unlearning result. The L2 implementation records that the Taylor recursion is heuristic for many momentum steps; no claim of exact trajectory replay is made.

An additional momentum-exact aggregate-step transfer applies the full constant-curvature heavy-ball recursion at each of the last W rounds, with optimizer velocity reset each round. At W=5, Bc=2048, it reached **74.42% CA / 2.17% ASR IID** and **69.83% / 2.88% non-IID**. This resolves the Taylor collapse but requires 790/935 GGN-vector products and 116.13/137.44 trillion GGN FLOPs respectively, on top of 14.70 trillion FLOPs for the L1 spectrum. It is therefore much more expensive than the default L1 and above FedEraser Ecal 1's client FLOPs. At Bc=512, W=5 it reached 73.79%/2.00% IID and 67.77%/3.47% non-IID at 29.03/34.36 trillion GGN FLOPs beyond the L1 spectrum. The aggregate-step transfer is exact for a shared quadratic and identical client K; non-IID heterogeneous K is rounded from its weighted mean, so it remains an approximation to FedAvg.

The registered rank/Bc L1 sweep's highest CA was rank 100, Bc 2048: 71.52% CA / 1.76% ASR IID, 51.71% / 7.17% non-IID. Three default L1 curvature/Lanczos seeds gave IID CA 70.27, 70.22, 69.82 and ASR 1.34, 1.30, 1.14; non-IID CA 43.07, 44.63, 47.71 and ASR 5.84, 8.58, 7.82. These are three stochastic curvature seeds on **one attacked trajectory seed per partition**, not three independently trained attacks.

The stable-transfer Bc=512 window sweep is complete:

| Partition | W=1 CA/ASR | W=2 CA/ASR | W=5 CA/ASR | W=10 CA/ASR |
|---|---:|---:|---:|---:|
| IID | 73.00/1.83 | 73.50/1.96 | 73.79/2.00 | 74.00/2.07 |
| Non-IID | 63.74/3.84 | 65.89/3.57 | 67.77/3.47 | 68.34/3.10 |

**P4 is inconclusive.** The fixed-spectrum contractions suggest a one-round stiff memory, yet non-IID ASR and CA continue changing through W=10 under the approximate aggregate-K transfer. IID ASR slightly worsens as W increases. A single attacked trajectory and aggregate-K approximation do not establish a general saturation horizon. Exact per-client heterogeneous local-step transfer, midpoint curvature and validation-selected model choice remain unverified. Stage 4's registered methods, tests, controls and plots are complete; the L2 Taylor failure and high cost of stable transfer are negative outcomes, not missing measurements.
