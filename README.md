# horizon ![travis](https://app.travis-ci.com/ADVRHumanoids/horizon.svg?branch=devel&status=passed)
A framework for trajectory optimization and optimal control for robotics based on CasADi

## Install
here we assume that a mambaforge python package manager is already installed in your system. if it is not the case follow the instruction [here](https://github.com/conda-forge/miniforge#mambaforge).
aftert installing mambaforge **assuming that you are in the project folder** you can create the environment for horizon by doing  

```
mamba env create -f environment.yml
conda activate hori
```
one important dependencies of horizon is **casadi-kin-dyn**. you need to install it in the environment that you have just created (all the casadi-kin-dyn dependencies are already satisfied in the newly created environment)
to install casadi-kin-dyn follow the instruction provided [here](https://github.com/ADVRHumanoids/casadi_kin_dyn/tree/collision)
after installing casadi-kin-dyn you can install horizon by doing

```
pip install -e .
```
### installing libhsl.so
to do 

## Documentations
Don't forget to check the [**documentation**](https://advrhumanoids.github.io/horizon/)!  
You will obtain hands-on details about the framework: a comprehensive documentation of the project, a collection of demonstrative videos and instructions to use Horizon in Docker.

## Try it!
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/FrancescoRuscelli/horizon-live/main?urlpath=lab/tree/index.ipynb)
