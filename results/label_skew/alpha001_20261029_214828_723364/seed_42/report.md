# Extreme label skew stress test

Dirichlet alpha = 0.01. See partition_report.json for realized skew.

| Model | CA % | ASR % | Untriggered target % | Macro F1 % | CA change pp |
|---|---:|---:|---:|---:|---:|
| attack | 33.48 | 99.97 | 1.19 | 26.30 | +0.00 |
| subtraction | 10.00 | 100.00 | 100.00 | 1.82 | -23.48 |
| kd | 10.00 | 100.00 | 100.00 | 1.82 | -23.48 |
| federaser_Ecal2 | 34.86 | 6.57 | 13.92 | 28.41 | +1.38 |

CA: 10,000 clean test images. ASR: 9,000 triggered non-target images.
Compare source ASR with its untriggered target rate to establish whether the backdoor learned.
Low ASR with collapsed CA does not establish useful unlearning. Inspect per-class recall in evaluation.json.
No counterfactual reference is trained; these results measure endpoint utility and backdoor behavior.
