# JVP-Cal continuation plan

Status checked 2026-11-16 in /home/teaching/isp on my-linux.

## Existing progress
- The existing checkout has the requested origin and branch jacobian-cal; no new commits for this work are present. docs/jacobian/ and results/jacobian/ are untracked and must be preserved.
- docs/jacobian/AUDIT.md exists, but its claims require independent verification, including its version comparison (2.9 is earlier than 2.14) and trajectory layout.
- IID attack output exists at results/jacobian/iid_attack_logged: 50 round directories and 500 client delta files. final_results.json records CA 74.9%, ASR 50.6111%, runtime 1394.13 seconds, seed 42. Compare with requested approximate 75% / 56%; investigate and report the ASR difference. Integrity and reproducibility remain to be checked.
- No stage DONE markers or scripts/proposed directory were found. No research process was visible at inspection; logs/attack_iid.pid exists but no matching active job was observed. No Non-IID trajectory, novelty check, estimator, tests, baselines, or final report were identified in the requested new locations. Verify the full repository before assuming anything is absent.
- The default python3 is 3.10.12 and cannot import torch. The existing audit reports a different Anaconda environment; locate and verify its interpreter and package versions before running. CIFAR-10 directory exists. GPU: RTX A5000, 24 GB. Filesystem reports 22 GB available, so check actual trajectory sizes and reserve space before further runs.

## Proceed in order after approval
1. Inspect repository instructions, prior work, logs and configurations; verify audit claims, environment, CIFAR contents, trajectory completeness and checkpoint compatibility. Correct the audit and record actual versions and deviations. Preserve existing files and avoid overwriting existing results.
2. Conduct and cite the novelty check, including the FedRecover paper and distinctions/limitations; preregister L0/L1/L2 before running evaluations.
3. Implement resumable, detached stage orchestration with logs, atomic checkpoints and DONE markers; ensure pre-existing outputs are validated before adopting them. Build and test curvature primitives, toy quadratic and MLP checks, and real-model smoke test. Address momentum, dropout, BN, weighting, round-one behavior and second-order approximation explicitly; report mathematical limitations honestly.
4. Measure one-round timing, FLOPs and storage, then estimate and cap the grid. Run same-trajectory IID baselines, oracle diagnostics and validation-only selection, followed by seeded test evaluations. Do not compare unmatched trajectories as if paired. Document checkpoint provenance for M_CF and M_sub; regenerate comparable references if feasible.
5. Repeat for Non-IID if resources permit. Implement and validate FedRecover-style L-BFGS against the paper; clearly flag any departures. Ensure all long jobs run detached with logs and checkpoint/resume support.
6. Produce plots, machine-readable results and report with exact commands, seeds, measured costs, skipped work and limitations. Evaluate L0/L1/L2 only against measured evidence. Commit new code, tests and documentation regularly without pushing; do not overwrite existing baseline scripts or results.

## Decisions and risks to resolve during work
- A finite second-order truncation is exact for a quadratic only in special cases (for example K <= 2); test a full recurrence for general K and the truncated expression against its correct analytic expansion.
- Momentum and stochastic mini-batches make the simple constant-Hessian formula approximate; measure rather than assume the effective learning rate.
- All long-running stages need genuine resumability, including baseline and trajectory generation; add wrappers/checkpoint support without modifying baseline scripts.
- A held-out split must be isolated before selection and must not leak test data. Estimate compute before selecting a manageable validation grid.
- Record per-run evidence and avoid unsupported FLOP or performance claims. If blocked, stop and report partial results.
