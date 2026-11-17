# Checkerboard unlearning results

Source run: `E:\01_Active_Projects\AI_and_Research\ISP-Client-Level-Backdoor-Unlearning-organised\results\checkerboard\noniid_20261024_150813_189`.
Method: training_free_rolling_tcug_2. Partition: noniid.

| Model | Clean accuracy % | Checkerboard ASR % |
|---|---:|---:|
| M_attack (source) | 71.14 | 80.89 |
| training_free_rolling_tcug_2 | 40.86 | 0.24 |

CA: 10,000 clean images. ASR: 9,000 triggered non-target images. Trigger: 4x4 bottom-right checkerboard.
No clean/reference training is performed. Existing weights are only evaluated if explicitly requested.
See run_config.json for exact source hashes and method settings. These files do not overwrite baseline tables.
