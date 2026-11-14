# Checkerboard unlearning results

Source run: `E:\01_Active_Projects\AI_and_Research\ISP-Client-Level-Backdoor-Unlearning\results\stress_tests\alpha001_20261029_214828_723364\seed_42\attack`.
Method: subtraction. Partition: noniid.

| Model | Clean accuracy % | Checkerboard ASR % |
|---|---:|---:|
| M_attack (source) | 33.48 | 99.97 |
| subtraction | 10.00 | 100.00 |

CA: 10,000 clean images. ASR: 9,000 triggered non-target images. Trigger: 4x4 bottom-right checkerboard.
No clean/reference training is performed. Existing weights are only evaluated if explicitly requested.
See run_config.json for exact source hashes and method settings. These files do not overwrite baseline tables.
