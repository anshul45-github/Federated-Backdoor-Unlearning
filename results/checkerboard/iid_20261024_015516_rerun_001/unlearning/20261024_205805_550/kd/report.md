# Checkerboard unlearning results

Source run: `E:\01_Active_Projects\AI_and_Research\ISP-Client-Level-Backdoor-Unlearning\results\attack_variants\checkerboard4x4\iid_20261024_015516_rerun_001`.
Method: kd. Partition: iid.

| Model | Clean accuracy % | Checkerboard ASR % |
|---|---:|---:|
| M_attack (source) | 75.26 | 92.18 |
| kd | 73.09 | 0.41 |

CA: 10,000 clean images. ASR: 9,000 triggered non-target images. Trigger: 4x4 bottom-right checkerboard.
No clean/reference training is performed. Existing weights are only evaluated if explicitly requested.
See run_config.json for exact source hashes and method settings. These files do not overwrite baseline tables.
