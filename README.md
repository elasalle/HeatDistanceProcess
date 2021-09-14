# Heat Distance Processes

This repository provides codes that generated the figure in the paper _INSERT PAPER'S NAME_.

## Summary

Heat Kernel Distance (HKD) and Heat Persistence Distance (HPD) are two notions of distance based on the comparison of heat diffusion on graphs. But instead of choosing a diffusion time _t_, we rather consider the whole process indexed by _t_ in [0,T] for some T>0. This allows to consider a multiscale notion of distance, which is supported by statistical properties.

More precisely, in the aforementioned parper, we prove that such process when computed on suitable weighted graphs satisfies a functional Cnetral Limit Theorem, as well as a Gaussian approximation. This allows to validate construction of consistent _confidence bands_ and consistent _two-sample tests_.

## Composition of the Repository

The files `extended_persistence.py`, `graphs.py`, `processes.py` contain tool functions to compute the different distance processes as well as implementation of the confidence bands and two-sample tests.

All files starting with the prefix `fig`, _i.e._ `fig_confidence_bands.py`, `fig_confidence_bands_HPD.py`,  `fig_tests.py`, `fig_tests_HPD.py` and `fig_neyman_pearson.py` allow to generate the figures included in the paper.

## How to run this code

### Libraries

This code was developed and tested with Windows 10 and `Python 3.8`.

It requires the following packages :
- `numpy`
- `scipy`
- `matplotlib`
- `networkx`
- `multiprocessing`
- `time`
- `gudhi`

### Guide using conda

We propose to use a new conda environment to run these codes.

```
conda create -n HeatDistProc python=3.8
conda activate HeatDistProc
python -m pip install matplotlib networkx gudhi scipy
python fig_confidence_bands.py
```
