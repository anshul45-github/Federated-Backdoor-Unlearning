# Conv2 + BatchNorm historical patch search

Source: `E:\01_Active_Projects\AI_and_Research\ISP-Client-Level-Backdoor-Unlearning\results\attack_variants\checkerboard4x4\iid_20261024_015516_rerun_001` (G50).
Selected: G29, alpha=0.25, ranking=single_channel_score, channels=[1, 19, 25, 41, 47, 52, 16, 18, 55, 21, 29, 0, 17, 22, 61, 37, 60, 44, 53, 35, 38, 26, 46, 4, 33, 7, 14, 15, 20, 28, 30, 31, 43, 54, 56, 59, 58, 51, 23, 12, 42, 11].
Selection CA-loss budget: 1.0 percentage points. Calibration accepted: False.

| Split/model | CA % | ASR % |
|---|---:|---:|
| Selection / original | 75.14 | 91.82 |
| Selection / chosen patch | 75.20 | 88.58 |
| Held-out report / original | 75.38 | 92.53 |
| Held-out report / chosen patch | 75.12 | 88.58 |

Channel scores, rankings, and prefixes use the selection split only. The report split is not used for choosing or revising the patch.
Random controls participate in selection; a random ranking can win. This search does not by itself establish superiority of score ranking.
These historical donors contain Client 0's pre-attack clean contributions. This is selective backdoor repair, not certified client-level unlearning.
