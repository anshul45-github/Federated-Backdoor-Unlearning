# Novelty check

FedEraser (Halimi et al., IWQoS 2021; https://doi.org/10.1109/IWQoS52092.2021.9521274 and https://arxiv.org/abs/2012.13891) uses retained historical updates with short local calibration and update norm rescaling.

FedRecover (Cao et al., https://arxiv.org/abs/2210.10936) is the closest prior work. It estimates client updates from historical information with L-BFGS and uses periodic exact corrections, abnormality handling, and final tuning. Therefore JVP-Cal should not be described as a new general paradigm for estimating client updates. The narrower contribution tested here is a matrix-free local Hessian/GGN Taylor correction for federated unlearning.

This repository prototype does not implement all FedRecover mechanisms. Its B2 comparison remains absent; a generic L-BFGS implementation would need to be called FedRecover-inspired unless secant construction, inverse-Hessian initialization, correction schedule, abnormality handling, and final tuning were matched to the paper. The reported experiment is limited to CIFAR-10, SmallCNN and simulated clients.
