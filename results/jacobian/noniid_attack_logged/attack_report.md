# Federated attack experiment

Experiment: `noniid_attack_logged`
Config: `configs/v2_noniid.yaml`. Output: `/home/teaching/isp/results/jacobian/noniid_attack_logged`.

| Final endpoint metric | Value |
|---|---:|
| Clean accuracy (CA), 10,000 clean images | 70.80% |
| Attack success rate (ASR), 9,000 triggered non-target images | 30.02% |
| Untriggered false-target rate, same 9,000 images without trigger | 3.83% |

Final metrics use the last training round; no test-selected checkpoint is substituted.

## Protocol

- Mode: attack; partition: noniid; seed: 42.
- Trigger: white_patch, 3 x 3, bottom right; target class 0.
- Malicious client: 0; 359 fixed poisoned samples out of 3597 local samples.
- Attack rounds: 30 through 50, inclusive.
- Clients: 10 with full participation; local epochs: 2; batch size: 64.
- SGD: learning rate 0.01, momentum 0.9, weight decay 0; no update amplification.
- Historical client-update logging enabled: True.

Peak attack-period ASR: 30.02% at round 50 (diagnostic only).

## Interpretation

Judge CA and final ASR together. A larger checkerboard is a candidate stronger trigger, not an established improvement until measured.
The old white-patch ASR is a different trigger evaluation. Existing clean/reference weights can be reused, but their ASR must be re-evaluated with this trigger for matched comparisons.
This run does not train clean/reference models or perform unlearning. Other legacy evaluators/unlearning launchers still assume the older trigger/paths and must be adapted before use.
