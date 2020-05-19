Deep Latent-Variable Kernel Learning
====

This is the python implementation of the model proposed in our paper [deep latent-variable kernel learning](https://arxiv.org/abs/2005.08467).

Deep kernel learning ([DKL](https://arxiv.org/abs/1511.02222)) leverages the connection between Gaussian process (GP) and neural networks (NN) to build an end-to-end, hybrid model. It combines the capability of NN to learn rich representations under massive data and the non-parametric property of GP to achieve automatic calibration. However, the deterministic encoder may weaken the model calibration of the following GP part, especially on small datasets, due to the free latent representation. 

We therefore present a complete deep latent-variable kernel learning (DLVKL) model wherein the latent variables perform **stochastic encoding** for regularized representation. Theoretical analysis however indicates that the DLVKL with \textit{i.i.d.} prior for latent variables suffers from **posterior collapse** and degenerates to a constant predictor. Hence, we further improve the DLVKL from two aspects: (i) the **complicated variational posterior through neural stochastic differential equations (NSDE)** to reduce the divergence gap, and (ii) the **hybrid prior taking knowledge from both the SDE prior and the posterior** to arrive at a flexible trade-off. Intensive experiments imply that the DLVKL-NSDE performs similarly to the well calibrated GP on small datasets, and outperforms existing deep GPs on large datasets.

The model is implemented based on [GPflow 1.3.0](https://github.com/GPflow/GPflow), [DiffGP](https://github.com/hegdepashupati/differential-dgp) and tested using Tensorflow 1.15.0. 

The demos for regression, binary classification, and dimensionality reduction are provided. The readers can run the demo files like

```
python demo_regression --model DLVKL
```
or
```
python demo_regression --model DLVKL_NSDE
```
