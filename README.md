# horizon ![travis](https://app.travis-ci.com/ADVRHumanoids/horizon.svg?branch=devel&status=passed)
A framework for trajectory optimization and optimal control for robotics based on CasADi

## Dependencies
- [`casadi`](https://github.com/casadi/casadi): built from source (currently not compatible with pip version)
    - required CMake options:
        -   -DWITH_PYTHON=ON
        -   -DWITH_PYTHON3=ON
- [`casadi_kin_dyn`](https://github.com/ADVRHumanoids/casadi_kin_dyn.git): built from source, or installed from pip


## Install

from source: (in the horizon folder) ```pip install .``` \
pip package: ```pip install casadi-horizon``` \
conda package: ```conda install horizon -c ftrancesco_ruscelli```

## Documentations
Don't forget to check the [**documentation**](https://advrhumanoids.github.io/horizon/)!  
You will obtain hands-on details about the framework: a comprehensive documentation of the project, a collection of demonstrative videos and instructions to use Horizon in Docker.

## Try it!
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/FrancescoRuscelli/horizon-live/main?urlpath=lab/tree/index.ipynb)
