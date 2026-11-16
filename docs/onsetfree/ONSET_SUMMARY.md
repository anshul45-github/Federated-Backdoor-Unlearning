# Stage 5 interim result: onset detector did not meet P3

The detector used rank-32 benign-update subspaces, leave-one-out benign residuals, robust z-scores and Page CUSUM. Thresholds were chosen from nine benign clients acting as pseudo-targets, with zero empirical false alarms on those nine traces. The true attack onset (round 30) was supplied only to compute the error and draw the evaluation line after detection.

| Partition | κ | Estimated onset | Error versus round 30 |
|---|---:|---:|---:|
| IID | 0.5 | 15 | -15 rounds |
| IID | 1.0 | 12 | -18 rounds |
| Non-IID | 0.5 | No alarm | Not defined |
| Non-IID | 1.0 | No alarm | Not defined |

**P3 is refuted for IID** under both registered κ values: the detector alarms well before the actual attack. The benign-only threshold does not protect against an already atypical malicious client before poisoning. The non-IID detector misses the onset entirely under the same calibration. These statements are based on the detector JSON and PNG under remote `results/onsetfree/onset/{iid,noniid}/`.

The IID soft-gating ablation is measured. With the detected onset at round 15 and no margin, `alpha=0` gave 69.82% CA / 0.44% ASR; `alpha=0.5` gave 68.74% / 0.31%; ungated `alpha=1` gave the default L0 result, 67.63% / 0.24%. Shifting the gate 10 rounds later gave 71.04% / 0.46% for `alpha=0`, which raises CA but remains before the true onset. The deliberately late +20 shift, added after seeing that the detector alarmed early, gave 72.96% / 1.16% for `alpha=0`, no margin. That +20 setting is **post-hoc and evaluation-informed**, so it cannot be counted as a valid trigger-blind or registered success. Non-IID had no detector alarm and therefore no valid gate. P3 remains refuted. The gate sweep is complete (27 IID variants evaluated). Threshold trade-off figures are in remote `results/onsetfree/plots/onset_threshold_tradeoff_{iid,noniid}.png`; the zero observed pseudo-target false alarms are only 0/9 empirical traces, not a population guarantee. Stage 5's experiment gate is complete, with a negative detector verdict and no deployable gate selected by independent benign validation.
