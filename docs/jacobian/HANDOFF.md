# CRITICAL UPDATE: 2026-11-16

The older handoff below contains invalid conclusions. The original JVP runner used historical client delta instead of replay-minus-old global displacement, and the run labelled fed_e1 actually did two epochs. These errors were corrected in commit 36db7d3. Do not use old JVP figures or the old plots and do not call iid_fed_e1 an Ecal=1 result. See results/jacobian/REPORT.md and corrected_seed_summary.json.

A detached corrected pipeline is running: PID in logs/corrected_pipeline.pid, output logs/corrected_pipeline.log. It runs corrected Ecal1/Ecal2 baselines for IID and Non-IID, sequentially, following exploratory corrected GGN runs. Each replay checkpoints per round. Check process and log before launching anything; do not duplicate jobs. The completed 3-seed results, earlier attack trajectories and tests are documented in the report. An oracle was stopped to avoid interfering with baseline timing; oracle_diagnostic.py now checkpoints each point, but must be launched after the pipeline, with a new output path. The script is not yet integrated into the master pipeline.

Remaining: validate baseline results, run corrected oracle and plots, implement train-mode BN and buffer recomputation, verified B2, FLOP accounting, validation-only tuning, replay stochastic state on resume, comprehensive ablations, and final report. Test-set metrics already inspected are exploratory, not suitable for hyperparameter selection. No L0/L1/L2 verdict can be asserted until same-trajectory Ecal1 completes, and no FLOP-speedup claim is supported. The audit, source and corrected interim results are committed; generated checkpoints/logs are not necessarily tracked. No push.

---

# JVP-Cal research handoff

Last updated: 2026-11-16 (Asia/Calcutta)
Remote workstation/repository: `/home/teaching/isp` on SSH alias `my-linux` (`teaching@172.18.40.113:2692`)
Branch: `jacobian-cal`
Latest known commit: `f5443e2` (`Prototype matrix-free JVP-Cal experiment`); later files are uncommitted. Do not push yet; user will ask separately. Do not request or accept GitHub passwords/private keys in chat. If later asked to push, inspect whether remote GitHub CLI/SSH auth is already configured and use that without exposing credentials.

## User's goal and constraints
The user asked to implement and honestly evaluate a Jacobian/Gauss-Newton replacement for FedEraser's client calibration, compare to existing repo results, finish remaining work, and later push/open a PR only when they ask. User explicitly says remote `/home/teaching/isp` is the workstation; do all coding, data, runs and outputs there. Keep progressing without repeatedly asking for confirmation. Long jobs (>~30 sec) must run detached with logs, PID, checkpoints/resume; poll briefly. No fabricated numbers. New outputs belong in `results/jacobian/`; do not overwrite existing baseline scripts/results. Work is on `jacobian-cal`; do not push until separately requested.

Original requested work: audit optimizer/round/retention/FedEraser/BN/metrics/trajectory/model; regenerate IID then Non-IID 50-round attack trajectories; run same-trajectory baselines; implement JVP/GGN and exact-HVP Taylor estimator with order 1/2, curvature choices, expansion point, Bc 64/256/1024, optional global FedEraser norm scaling, anchors m=infinity/10/5/3 and warmup 0/2, BN buffer options; baselines stale, FedEraser Ecal 1/2, FedRecover-like L-BFGS, M_CF, M_sub; validations; oracle diagnostics; validation-only tuning; multi-seed metrics, FLOPs/wall time/storage; plots/report; master `scripts/proposed/run_all.sh`; regular commits; final report with L0/L1/L2 verdict. Exact original prompt is in the chat history.

## Remote environment and evidence
- Python 3.12.9, PyTorch 2.9.0+cu128, torchvision 0.24.0+cu128, CUDA available, RTX A5000 24 GB.
- README pins Python 3.11.6 / PyTorch 2.14.0; actual environment deviates.
- CIFAR-10 exists at `data/cifar-10-batches-py`.
- Disk had 22 GB free before Non-IID trajectory; last seen ~20 GB free.
- Original attack audit at `docs/jacobian/AUDIT.md`; verify/correct if necessary. FedEraser: SGD lr .01, momentum .9, wd 0, batch64, E=2, 50 rounds, 10/10 clients; retained rounds 1..50. FedEraser skips calibration on round1; rounds 2..50 local calibrate and globally rescale parameter-vector norm; BN running stats unscaled; `num_batches_tracked` bypassed. CIFAR SmallCNN 545,290 trainable params. CA full 10k, ASR 9k non-target images with 3x3 white bottom-right patch to class0.
- Existing historical phase2 FedEraser/M_CF/M_sub summaries exist, but historical source trajectories are absent, so they are not verified paired comparisons. Cite as historical/context only. `results/phase2/federaser/federaser_comparison.md` has legacy comparison table.

## Work completed
- IID trajectory: `results/jacobian/iid_attack_logged/trajectory`, 500 delta files; attack JSON records CA 74.9%, ASR 50.6111%, runtime 1394.13 s.
- Non-IID trajectory regenerated at `results/jacobian/noniid_attack_logged/trajectory`, 500 deltas; attack JSON CA 70.8%, ASR 30.0222%, runtime 1370.89 s.
- Curvature helpers: `src/unlearning/jvp_cal.py`: flat/unflat utilities, softmax GGN JVP/VJP, exact HVP double-backward, Taylor update, global norm rescaling.
- Tests: `tests/test_jvp_cal.py` 2 passed (GGN vs explicit Jacobian; exact HVP vs finite differences).
- Runner: `scripts/proposed/run_jvp_cal.py`, round checkpoint and `progress.json` resume; supports `jvp`, `stale`, `fed_e1`, GGN/exact, order, Bc, rescaling. Config `configs/jvp_cal_iid.yaml`.
- `scripts/proposed/run_all.sh` exists but is incomplete: only attack presence checks, stale/JVP runs, tests; lacks many requested stages and robust markers.
- `scripts/proposed/oracle_diagnostic.py` sampled IID rounds 10/25/40, clients 1/5/9; output `results/jacobian/iid_oracle_sample.json`.
- `docs/jacobian/NOVELTY_CHECK.md`: FedRecover (arXiv:2210.10936) is closest prior; don't claim broad update-estimation novelty.
- `results/jacobian/REPORT.md` draft and plots `pareto_ca_runtime.png`, `iid_delta_growth.png`, `noniid_delta_growth.png`; metric/drift JSON present.
- Commit f5443e2 was created with one-command author identity Codex / codex@openai.com. Subsequent additions currently uncommitted. Check `git status`.

## Measured results (all numbers from result JSON / historical repo files)
| Setting | Method | CA % | ASR % | runtime s |
|---|---|---:|---:|---:|
| IID | Attack | 74.90 | 50.61 | 1394.13 |
| IID | stale | 67.08 | 0.4667 | 1.4466 |
| IID | FedEraser Ecal1 replayed on this trajectory | 75.13 | 2.60 | 949.87 |
| IID | JVP GGN order1 unscaled | 63.28 | .4556 | 9.750 |
| IID | JVP GGN order1 scaled seed123 | 63.27 | .4556 | 9.903 |
| IID | JVP scaled seed124 | 62.13 | .40 | 10.034 |
| IID | JVP scaled seed125 | 62.85 | .4667 | 10.022 |
| IID | JVP exact-Hessian order1 scaled seed123 | 66.20 | .5222 | 11.181 |
| IID | JVP GGN order2 scaled | See `results/jacobian/iid_jvp_g64_o2/metrics.json` |
| Non-IID | Attack | 70.80 | 30.0222 | 1370.894 |
| Non-IID | stale | 31.44 | 2.6556 | 1.4574 |
| Non-IID | JVP GGN order1 scaled seed123 | 10.27 | 0 | 9.949 |

Historical comparison for context only: IID FedEraser Ecal2 74.99/2.3444, M_CF 74.87/3.0111; repository summary has M_sub 66.88/.24 IID and Non-IID M_sub 28.04/.87, Non-IID M_CF 69.64/3.40; provenance/source checkpoints for these historical results need investigation and must not be represented as paired. Check source files before quoting.

The current prototype uses `steps=1`, not the true ~158 stochastic momentum SGD steps/client/round. Functional curvature currently uses model eval mode (to avoid BatchNorm in-place mutation under torch.func), deviating from requested train-mode curvature. These are major limitations. IID oracle's first displayed selected point showed negligible JVP improvement: cosine .10346 vs stale .10250; relative error 1.29493 vs 1.29542; displacement ratio .1265. Inspect all JSON rows and plot it before reporting an aggregate. Oracle exact updates train over the entire client partition, whereas curvature uses 8 samples; marked caveat in JSON.

Current verdict: negative. IID JVP gives much lower ASR but CA loss (fails L0, thus L1/L2); Non-IID JVP has severe CA collapse (10.27%) and loses to stale by 21.17 points. No FLOPs measured, no FLOP equivalence claims.

## Remaining work (next chat should continue, not restart)
1. Inspect current `git status`, exact ablation output files, and run logs; update report table with second-order result and all oracle rows. Plot oracle cosine/error by round and Delta/theta growth. Make sure current plots/data reflect all measured variants.
2. Fix/complete implementation correctness: runner currently lacks expansion-point old/midpoint choice, anchor schedule m/w, BN buffer choices, correct SGD momentum/effective-step approximation, and robust YAML config loading. Add a true resumed master pipeline with stage markers and correct wrappers. Verify stage skip only when completion artifact validated.
3. Complete meaningful tests: toy quadratic update recurrence (the initial user's formula's truncated second order is not exact for arbitrary K; clarify/test small K), toy MLP GGN/exact tests exist; add model smoke/BN checks.
4. Run Bc/order/curvature/rescale ablation using existing small script `scripts/proposed/run_small_ablation.sh`; inspect its completion and logs. No validation split-based hyperparameter search has been performed.
5. Implement a correctly labeled FedRecover-inspired L-BFGS baseline only if faithful mechanics can be verified from paper; otherwise record explicitly as skipped and why. Do not falsely call generic LBFGS FedRecover.
6. Evaluate historical M_sub/M_CF checkpoints if available; no checkpoint was found in prior checked paths. Search remaining results for final model files and provenance. Existing summary tables can be historical context only.
7. FLOP counter/analytic estimate and server storage/compute still absent. Wall clock only.
8. Fix report inconsistencies/wording and add exact commands. Report all noncompliance with spec, no fabricated metrics.
9. Run final tests, inspect git diff, commit remaining changes locally on jacobian-cal. Do not push until user asks. User said they will ask to push later.

## Useful commands
SSH command prefix from the Windows Codex environment:
`ssh -F 'C:\Users\hp\.ssh\config' my-linux 'cd /home/teaching/isp && ...'`
Interpreter `/home/teaching/miniconda3/bin/python`; export `PYTHONPATH=/home/teaching/isp`.
Run tests: `/home/teaching/miniconda3/bin/python -m pytest -q tests/test_jvp_cal.py`.
Check jobs: `pgrep -af 'run_jvp_cal|oracle_diagnostic|run_experiment_v2'`; logs in repo `logs/`; output JSON in `results/jacobian/`.
