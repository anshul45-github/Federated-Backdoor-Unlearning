# L1/L2 interim results on the new white-patch trajectories

The registered default L1 uses rank 50, 2,048 retained-client examples, eval-mode BatchNorm and a momentum-aware heavy-ball transfer. It ran without trigger, target, or attack-onset metadata. The IID endpoint reached **70.27% CA and 1.34% ASR**, versus 67.63%/0.24% for default L0, 64.31%/0.11% for C1 and 71.99%/2.33% for matched FedEraser Ecal 1. The L1 computation used 100 GGN-vector products over 32 batches, counted as 14,699,986,944,000 forward/backward-equivalent ATen FLOPs; matched IID FedEraser Ecal 1 used 77,183,608,320,000 client FLOPs. The ratio is about 5.25, below the prespecified 10x S1 saving.

On a fixed 512-example benign sample disjoint from the L1 curvature sample, C1 increased cross-entropy by 0.2733 and had a Hessian quadratic of 0.3986. L1 increased it by 0.0909 and had a quadratic of 0.1567. Default L0 was between them (0.1570 actual, 0.2785 quadratic). This supports P1 for IID on this sample; the non-IID validation is pending. The sample is disjoint from curvature estimation but drawn from original training clients, so it is not a genuinely held-out population.

IID whole-vector cosine to the evaluation-only counterfactual displacement was 0.0834 for L1, versus 0.2353 for C1 and 0.1803 for L0. Among the 10 layers with all three cosines defined, L1 beat both C1 and L0 in only 2. Two L0 bias-layer displacements were zero, so their cosines are undefined. **P2 is refuted for IID**, both globally and by a majority of comparable layers. Non-IID remains pending.

All 50 registered IID L1 eigenmodes had contraction below 0.5; the maximum recorded contraction was 0.0002802. Under this fixed-curvature heavy-ball approximation, stiff modes have effectively a one-round memory. The observed L2 Taylor collapse prevents validating a W saturation claim against a stable time-varying method.

The registered IID L2 first-order Taylor window sweep gave approximately **10% CA** at every W in {1,2,5,10} and Bc in {512,2048}; ASR varied between 0% and 100%. This is a failed, unstable approximation under the recorded `eta_eff=eta/(1-momentum)` and weighted local step count. It must not be treated as a successful unlearning result. P4 cannot be confirmed from this sweep. The L2 implementation records that the Taylor recursion is heuristic for many momentum steps; no claim of exact trajectory replay is made.

Pending: non-IID L1/L2, three L1 curvature seeds, rank/sample sweeps, Hessian and BatchNorm ablations, and a stable finite-horizon L2 transfer. No `stage4.DONE` marker is asserted.
