# JVP-Cal Experiment Report

## Verdict
Measured IID first-order GGN lowers ASR but loses clean accuracy against stale reuse and paired FedEraser E_cal=1. It fails L0 (63.27% CA versus 75.13%, loss 11.86 points), therefore also fails L1/L2. This is a negative result for this prototype configuration.

## IID results
| Method | CA % | ASR % | Runtime s | Provenance |
|---|---:|---:|---:|---|
| M_attack | 74.90 | 50.61 | 1394.13 | IID attack JSON |
| Stale reuse | 67.08 | 0.47 | 1.45 | same stored IID trajectory |
| FedEraser E_cal=1 | 75.13 | 2.60 | 949.87 | replayed on same trajectory |
| FedEraser E_cal=2 | 74.99 | 2.34 | 629.82* | historical repo result |
| JVP-Cal GGN, order 1, no scaling | 63.28 | 0.46 | 9.75 | same trajectory |
| JVP-Cal GGN, order 1, global norm scaling | 63.27 | 0.46 | 9.90 | same trajectory |
| M_CF | 74.87 | 3.01 | 2513.87* | historical repo result |

*Historical rows lack retained source trajectories and are contextual, not paired. FLOPs were not measured; no FLOP claim is made.

## Method and limitations
The code uses matrix-free softmax GGN products via torch.func JVP/VJP and exact Hessian-vector products via double backward. Tests compare GGN with explicit Jacobian construction and exact HVP with finite differences (2 passed). The IID run used Bc=64, one estimator seed (123), order 1 and configured steps=1. This does not model roughly 158 stochastic momentum SGD steps/client-round. Curvature uses eval-mode functional buffers because train-mode BatchNorm mutates counters under torch.func, deviating from the requested train-mode convention. The implementation is a prototype diagnostic.

FedRecover (https://arxiv.org/abs/2210.10936) already estimates client updates with L-BFGS and periodic corrections; no broad novelty claim is made. See docs/jacobian/NOVELTY_CHECK.md.

Environment: Python 3.12.9, PyTorch 2.9.0+cu128, torchvision 0.24.0+cu128. CIFAR-10 and the IID 500-delta trajectory are present.

## Outstanding
Non-IID regeneration is running detached. Non-IID JVP-Cal, oracle exact-update diagnostics, Taylor validity plot, broad ablation, FedRecover-faithful L-BFGS, M_sub checkpoint evaluation, FLOP counts, and three-seed statistics remain incomplete. Results are limited to CIFAR-10/SmallCNN with simulated clients.

## Reproduction
From repo root, set PYTHONPATH=. and use /home/teaching/miniconda3/bin/python scripts/proposed/run_jvp_cal.py with distinct --out-dir values. Controls: --method stale and --method fed_e1. Tests: python -m pytest -q tests/test_jvp_cal.py. Long runs were detached with logs under logs/.

## Non-IID results (newly completed)
The Non-IID attack trajectory was regenerated with seed 42 and contains 500 client deltas. Its attack result is 70.80% CA, 30.02% ASR, runtime 1370.89 s. Rescaled JVP-Cal (GGN, Bc=64, order 1, seed 123) gives 10.27% CA and 0.00% ASR in 9.95 s. This is a severe utility collapse. Stale reuse is being run on this same trajectory to distinguish estimator-specific damage from subtraction/replay behavior.
Non-IID same-trajectory control: stale reuse is 31.44% CA / 2.66% ASR in 1.46 s, while JVP-Cal is 10.27% / 0.00%. Thus JVP-Cal loses 21.17 clean-accuracy points relative to stale reuse on Non-IID as well.

## Generated figures and trajectory diagnostics
The report companion plots are `pareto_ca_runtime.png`, `iid_delta_growth.png`, and `noniid_delta_growth.png`; source numeric data are in `measured_metrics.json`, `iid_delta_growth.json`, and `noniid_delta_growth.json`. Delta growth is measured against each attack trajectory's initial model. The requested round/client cosine and relative-error oracle plot has not been computed because the prototype runner does not yet expose per-client curvature estimates and exact local counterfactual updates; no values are fabricated. Client FLOPs also remain unmeasured. Reported wall times are JSON run times, not FLOP estimates.
