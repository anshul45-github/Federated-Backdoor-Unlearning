# L1/L2 interim results on the new white-patch trajectories

The registered default L1 uses rank 50, 2,048 retained-client examples, eval-mode BatchNorm and a momentum-aware heavy-ball transfer. It ran without trigger, target, or attack-onset metadata. The IID endpoint reached **70.27% CA and 1.34% ASR**, versus 67.63%/0.24% for default L0, 64.31%/0.11% for C1 and 71.99%/2.33% for matched FedEraser Ecal 1. The L1 computation used 100 GGN-vector products over 32 batches, counted as 14,699,986,944,000 forward/backward-equivalent ATen FLOPs; matched IID FedEraser Ecal 1 used 77,183,608,320,000 client FLOPs. The ratio is about 5.25, below the prespecified 10x S1 saving.

The same non-IID default L1 reached **43.07% CA / 5.84% ASR**, far below matched FedEraser Ecal 1's 67.69% CA / 2.51% ASR and below its own attacked model's 71.59% CA. All 50 non-IID stiff modes had contraction below 0.5 (maximum 0.001457). This is a negative result for the default L1 outside IID.

On fixed 512-example benign samples disjoint from the L1 curvature samples, C1 increased cross-entropy by 0.2733 IID and 6.7303 non-IID, whereas L1 increased it by 0.0909 and 0.9720. The Hessian quadratics were 0.3986/7.6166 for C1 versus 0.1567/0.4680 for L1. This supports P1 on both measured samples, despite the poor non-IID test CA of L1. The samples are disjoint from curvature estimation but drawn from original training clients, so they are not genuinely held-out populations.

IID whole-vector cosine to the evaluation-only counterfactual displacement was 0.0834 for L1, versus 0.2353 for C1 and 0.1803 for L0. Non-IID whole-vector cosines were 0.0125, 0.1211 and 0.1498. In each partition, among the 10 layers with all three cosines defined, L1 beat both C1 and L0 in only 2. Two L0 bias-layer displacements were zero, so their cosines are undefined. **P2 is refuted on both measured trajectories**, globally and by a majority of comparable layers.

All 50 registered IID L1 eigenmodes had contraction below 0.5; the maximum recorded contraction was 0.0002802. Under this fixed-curvature heavy-ball approximation, stiff modes have effectively a one-round memory. The observed L2 Taylor collapse prevents validating a W saturation claim against a stable time-varying method.

In the matched IID rank-20, 512-example curvature ablation, exact Hessian gave 68.74% CA / 1.32% ASR and GGN gave 68.84% / 1.33%. The result is similar at this registered rank/sample setting, but does not establish equivalence at rank 50 or for non-IID.

The registered IID L2 first-order Taylor window sweep gave approximately **10% CA** at every W in {1,2,5,10} and Bc in {512,2048}; ASR varied between 0% and 100%. This is a failed, unstable approximation under the recorded `eta_eff=eta/(1-momentum)` and weighted local step count. It must not be treated as a successful unlearning result. P4 cannot be confirmed from this sweep. The L2 implementation records that the Taylor recursion is heuristic for many momentum steps; no claim of exact trajectory replay is made.

Pending: non-IID L2, three L1 curvature seeds, complete rank/sample sweeps, non-IID Hessian and BatchNorm ablations, and a stable finite-horizon L2 transfer. No `stage4.DONE` marker is asserted.
