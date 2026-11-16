# Stage 5 interim result: onset detector did not meet P3

The detector used rank-32 benign-update subspaces, leave-one-out benign residuals, robust z-scores and Page CUSUM. Thresholds were chosen from nine benign clients acting as pseudo-targets, with zero empirical false alarms on those nine traces. The true attack onset (round 30) was supplied only to compute the error and draw the evaluation line after detection.

| Partition | κ | Estimated onset | Error versus round 30 |
|---|---:|---:|---:|
| IID | 0.5 | 15 | -15 rounds |
| IID | 1.0 | 12 | -18 rounds |
| Non-IID | 0.5 | No alarm | Not defined |
| Non-IID | 1.0 | No alarm | Not defined |

**P3 is refuted for IID** under both registered κ values: the detector alarms well before the actual attack. The benign-only threshold does not protect against an already atypical malicious client before poisoning. The non-IID detector misses the onset entirely under the same calibration. These statements are based on the detector JSON and PNG under remote `results/onsetfree/onset/{iid,noniid}/`.

The optional soft-gating CA/ASR ablation is still pending. Because onset estimates are poor, a gate must be treated as a potentially harmful ablation, not as a default method. No `stage5.DONE` is asserted.
