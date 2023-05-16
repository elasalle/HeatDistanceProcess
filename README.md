# Heat Distance Processes

This repository provides codes that generated the figures in the paper [Heat diffusion distance processes: a statistically founded method to analyze graph data sets](https://arxiv.org/abs/2109.13213).

## Summary

Heat Kernel Distance (HKD) and Heat Persistence Distance (HPD) are two notions of distance based on the comparison of heat diffusion on graphs. But instead of choosing a diffusion time _t_, we rather consider the whole process indexed by _t_ in [0,T] for some T>0. This allows to consider a multiscale notion of distance, which is supported by statistical properties.

More precisely, in the aforementioned paper, we prove that such process when computed on suitable weighted graphs satisfies a functional Central Limit Theorem, as well as a Gaussian approximation. This allows to validate constructions of consistent _confidence bands_ and consistent _two-sample tests_.

## Composition of the Repository

The files `extended_persistence.py`, `graphs.py`, `processes.py` contain tool functions to compute the different distance processes as well as implementation of the confidence bands and two-sample tests. `kernel_two_sample_test.py` allows to compute the MMD-based tests.

All files starting with the prefix `fig` allow to generate the figures included in the paper.

- `fig_confidence_bands.py` : confidence bands for HKD. (Figure 2)
- `fig_confidence_bands_HPD.py` : confidence bands for HPD. (Figure 3 and 4)
- `fig_tests.py` : test performances with HKD. (Figure 5)
- `fig_tests_HPD.py` : test performances with HPD. (Figure 6)
- `fig_tests_HPD_random_size.py` : test performances with HPD for graphs with random sizes. (Figure 7)
- `fig_tests_vs_others.py` : comparison of the HKD-based test with other methods (Figure 8a)
- `fig_tests_vs_others_WS.py` : comparison of the HKD-based test with other methods, using the Watts-Strogatz model (Figure 8b)
-  `fig_tests_HPD_random_size_vs_kernel.py` : comparison of the HPD-based test with MMD-based tests. (Figure 9)
- `fig_neyman_pearson.py` : test performance for HKD, when working close to the Neyman-Pearson phase transition. (Figure 10)

Note that the files to evaluate tests' performances may take from a few minutes up to a few hours (especially with HPD) to compute.

## How to run this code

This code was developed and tested with Windows 10 and `Python 3.8`.

### Libraries

It requires the following packages :
- `numpy`
- `scipy`
- `matplotlib`
- `networkx`
- `multiprocessing`
- `time`
- `gudhi`
- `grakel`
- `sklearn`

### Guide using conda

We propose to use a new conda environment to run these codes.

Open a command prompt, go to the repository and type the following commands :

```
conda create -n HeatDistProc python=3.8
conda activate HeatDistProc
python -m pip install matplotlib networkx gudhi scipy grakel scikit-learn
python fig_confidence_bands.py
```
