# Checkerboard unlearning results

Source run: `/DATA/teaching/tcug/results/checkerboard/noniid_20261024_150813_189`.
Method: training_free_tcug_2. Partition: noniid.

| Model | Clean accuracy % | Checkerboard ASR % |
|---|---:|---:|
| M_attack (source) | 71.14 | 80.89 |
| training_free_tcug_2 | 68.14 | 0.16 |

CA: 10,000 clean images. ASR: 9,000 triggered non-target images. Trigger: 4x4 bottom-right checkerboard.
No clean/reference training is performed. Existing weights are only evaluated if explicitly requested.
See run_config.json for exact source hashes and method settings. These files do not overwrite baseline tables.
