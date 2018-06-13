# Tension Factor
**Bayesian-evidence based measure for quantifying tension between model parameters obtained from two experimental datasets**

This repository contains some of the code and scripts used to calculate the evidences to obtain the new statistical measure for the papaer [`arXiv:1806.04292`](https://arxiv.org/abs/1806.04292).

The minimization is done using [`scipy.optimize.differential_evolution`](https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.optimize.differential_evolution.html) and the Bayesian evidence is calculated using [`pymultinest`](https://github.com/JohannesBuchner/PyMultiNest)
