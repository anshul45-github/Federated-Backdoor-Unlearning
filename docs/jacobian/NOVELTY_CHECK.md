# Literature and credibility check

- [FedEraser, Liu et al.](https://arxiv.org/abs/2012.13891): historical client updates and calibration; the earlier audit incorrectly attributed the paper to Halimi. The arXiv abstract calls the work *Federated Unlearning*; its conference title is *FedEraser*.
- [FedRecover, Cao et al.](https://arxiv.org/abs/2210.10936): historical information estimates updates for recovery, and the paper describes limited-memory BFGS and exact correction mechanisms. This is the closest prior art; estimating updates from history with Taylor/Hessian approximations is not novel.
- [FedRecovery, Zhang et al.](https://ieeexplore.ieee.org/document/10189868/): the indexed abstract describes weighted gradient residuals and Gaussian noise. The IEEE page could not be opened during this check, so its detailed claims remain unverified.
- [FURR, Liu et al.](https://arxiv.org/abs/2203.07320): the abstract verifies a rapid retraining method. Its exact diagonal empirical-Fisher implementation requires full-paper verification.
- [2F2L / FedRemoval](https://arxiv.org/abs/2306.02216): the abstract verifies linearized federated training and certified removal. The stated use of public-server-data Hessian approximation was not verified from the abstract.
- [DeltaGrad, Wu et al.](https://proceedings.mlr.press/v119/wu20b.html): cached training history enables rapid retraining. The full paper describes quasi-Newton ideas.
- [Crab](https://arxiv.org/abs/2401.08216): an additional poisoning-recovery paper, relevant to correction and certified recovery.

The narrow experiment is matrix-free GGN/JVP or Hessian corrections combined with this repository's globally norm-rescaled FedEraser replay, compared in the same codebase. This is not a claim of methodological novelty. A faithful FedRecover baseline has not been implemented, so no comparison to it is claimed.
