# Stage 2: abstract-level novelty check

Checked 2026-11-16 against the linked primary abstracts. "Onset" below means the attack activation round; most papers are general client unlearning and do not frame that variable. Abstract silence is **not** proof of independence from onset. This is a targeted abstract review, not a full-text priority or patent search.

| Work | Verified abstract-level contribution | Attack-onset requirement visible in abstract? |
|---|---|---|
| [FedEraser](https://arxiv.org/abs/2012.13891) (2020/21) | Reconstructs the unlearned FL model from stored client updates with calibration. | No explicit onset requirement. |
| [FedRecover](https://arxiv.org/abs/2210.10936) (2022/23) | Estimates client updates using stored global models and per-round updates, with exact-update correction phases. | No explicit onset requirement. Closest stored-trajectory recovery prior art. |
| [Crab](https://arxiv.org/abs/2401.08216) (2024) | Selective history storage and adaptive rollback to a historical model not significantly affected by malicious clients. | No known onset stated, but it chooses a rollback point from influence evidence. |
| [Santi et al.](https://arxiv.org/abs/2510.13606) (2025) | Task arithmetic plus neural tangent kernel for rapid client influence removal. | Not stated. Abstract does not establish the PDF's claim about standalone task vectors or training-time changes. |
| [HF-KCU](https://arxiv.org/abs/2605.20341) (2026) | Krylov/conjugate-gradient approximation to influence-function client removal under adversarial perturbations. | Not stated. |
| [Certified decentralized unlearning](https://arxiv.org/abs/2601.06436) (2026) | Tracks client influence propagation in decentralized FL and applies Fisher/Newton-style correction with noise. | Not stated. |
| [TracIn](https://arxiv.org/abs/2002.08484) (2020) | First-order training-example influence using checkpoints and gradients. | Not an attack-onset method. |
| [FURR](https://arxiv.org/abs/2203.07320) (2022) | Client-side rapid retraining for federated forgetting. | Not stated. **Correction:** its abstract does not describe diagonal-Fisher Newton updates as the proposal PDF claims. |
| [SpectralKrum](https://arxiv.org/abs/2512.11760) (2025) | Learns a low-dimensional benign historical-update subspace and filters orthogonal residuals for Byzantine defense. | Not an unlearning or onset-estimation method. |
| [Wu, Zhu, Mitra](https://mlanthology.org/iclrw/2023/wu2023iclrw-unlearning/) (2023) | Subtracts accumulated attacker updates, then restores performance with knowledge distillation. | No explicit onset requirement. Direct C1 prior art. |
| [Gur-Ari et al.](https://arxiv.org/abs/1812.04754) (2018) | Observes gradients concentrating in a small Hessian-eigenvector subspace. | Not an FL unlearning method. |

Additional directly relevant papers found and opened:

| Work | Verified abstract-level contribution | Attack-onset requirement visible in abstract? |
|---|---|---|
| [Subspace based Federated Unlearning (SFU)](https://arxiv.org/abs/2302.12448) (2023) | Removes target-client influence by updating in the orthogonal complement of other clients' input-gradient subspace. This already publishes a benign-preserving orthogonal-subspace idea, though it builds the subspace differently from stacked stored deltas. | Not stated. |
| [SIFU](https://proceedings.mlr.press/v238/fraboni24a.html) (2024) | Sequential informed client unlearning for FedAvg with formal guarantees and no extra client computation, per abstract. | Not stated. |
| [FLDetector](https://arxiv.org/abs/2207.09209) (2022) | Detects inconsistent client updates across iterations using historical-update predictions from L-BFGS. | It detects behavior over time, but its abstract does not claim a CUSUM onset estimate. |
| [FedCARE](https://ijcai.org/proceedings/2026/621) (2026) | Uses conflict-aware projection with gradient-ascent forgetting and recovery. This adds another prior art for projection to preserve utility, but requires local forget-data access in its stated method. | Not stated. |

## Verdict on the four proposed components

1. **Contrast forcing `p_m(U_m - B)`:** the expression follows directly from renormalized FedAvg subtraction. Do not claim the algebra itself as novel. FedEraser and FedRecover already use stored histories; Wu et al. publish attacker-update subtraction. Whether this exact contrast has been previously written is **unverified** by abstracts.
2. **Finite-horizon spectral kernel:** checkpoint-summed influence (TracIn), history-based recovery (FedRecover), and Hessian/Krylov influence correction (HF-KCU) are prior art. The particular per-round contraction kernel in the PDF was **not established as previously published or novel** from abstracts alone. Full-text mathematical comparison remains necessary.
3. **Stiff/flat split and orthogonal projection:** low-dimensional gradient structure (Gur-Ari), benign-update subspaces (SpectralKrum), and orthogonal-subspace federated unlearning (SFU) are published. The broad projection concept is **not novel**. The precise stacked-delta, uncentered L0 estimator may differ, but that distinction needs full-text checks and empirical value.
4. **Leave-one-out CUSUM:** temporal client-update anomaly detection is published (FLDetector); the specific leave-one-out robust residual plus Page CUSUM combination was **not verified** in this abstract search. Treat Stage 5 as an ablation, not a novelty claim.

The proposal's overall combination is a research hypothesis. Neither originality nor effectiveness has been established by this check. The withdrawn [arXiv:2508.19065](https://arxiv.org/abs/2508.19065) appeared in search results and is excluded from the evidence above.
