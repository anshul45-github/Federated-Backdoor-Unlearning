# Federated attack experiment

Experiment: `checkerboard4_extreme_noniid_alpha001_seed42`
Config: `E:\01_Active_Projects\AI_and_Research\ISP-Client-Level-Backdoor-Unlearning\results\stress_tests\alpha001_20261029_214828_723364\seed_42\config.yaml`. Output: `E:\01_Active_Projects\AI_and_Research\ISP-Client-Level-Backdoor-Unlearning\results\stress_tests\alpha001_20261029_214828_723364\seed_42\attack`.

| Final endpoint metric | Value |
|---|---:|
| Clean accuracy (CA), 10,000 clean images | 33.48% |
| Attack success rate (ASR), 9,000 triggered non-target images | 99.97% |
| Untriggered false-target rate, same 9,000 images without trigger | 1.19% |

Final metrics use the last training round; no test-selected checkpoint is substituted.

## Protocol

- Mode: attack; partition: noniid; seed: 42.
- Trigger: checkerboard, 4 x 4, bottom right; target class 0.
- Malicious client: 0; 999 fixed poisoned samples out of 9999 local samples.
- Attack rounds: 30 through 50, inclusive.
- Clients: 10 with full participation; local epochs: 2; batch size: 64.
- SGD: learning rate 0.01, momentum 0.9, weight decay 0; no update amplification.
- Historical client-update logging enabled: True.

Raw patch pixels (W = white 1.0, B = black 0.0; identical pattern in all RGB channels):

```text
W B W B
B W B W
W B W B
B W B W
```

Both colors are normalized using the CIFAR-10 mean/std before insertion.

Peak attack-period ASR: 99.98% at round 40 (diagnostic only).

## Interpretation

Judge CA and final ASR together. A larger checkerboard is a candidate stronger trigger, not an established improvement until measured.
The old white-patch ASR is a different trigger evaluation. Existing clean/reference weights can be reused, but their ASR must be re-evaluated with this trigger for matched comparisons.
This run does not train clean/reference models or perform unlearning. Other legacy evaluators/unlearning launchers still assume the older trigger/paths and must be adapted before use.
