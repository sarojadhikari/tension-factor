# Dataset-evidence ratio
Bayesian evidence ratio for comparing two datasets in cosmology. 

This repository contains some some the code and scripts used to calculate the evidences to obtain the dataset-evidence ratio for the papaer `arXiv:180x.xxxxx`.

The minimization is done using [`scipy.optimize.differential_evolution`](https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.optimize.differential_evolution.html) and the Bayesian evidence is calculated using [`pymultinest`](https://github.com/JohannesBuchner/PyMultiNest)
