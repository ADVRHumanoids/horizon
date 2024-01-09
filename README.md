# horizon ![travis](https://app.travis-ci.com/ADVRHumanoids/horizon.svg?branch=devel&status=passed)
A framework for trajectory optimization and optimal control for robotics based on CasADi

## Dependencies
- [`casadi`](https://github.com/casadi/casadi): built from source (currently not compatible with pip version)
    - required CMake options:
        -   -DWITH_PYTHON=ON
        -   -DWITH_PYTHON3=ON
- [`casadi_kin_dyn`](https://github.com/ADVRHumanoids/casadi_kin_dyn.git): built from source, or installed from pip


## Install

### Suggested way:

We suggest to use the `forest` tool to install the main dependencies of the controller more easily:
```
[sudo] pip3 install hhcm-forest
mkdir forest_ws && cd forest_ws
forest init
echo ". ~/forest_ws/setup.bash" >> ~/.bashrc
forest add-recipes git@github.com:advrhumanoids/multidof_recipes.git
```
Once `forest` has been sucessfully installed, you can now install horizon:
```
cd ~/forest_ws
forest grow horizon
```

### Other options:
from source: (in the horizon folder) ```pip install . --no-deps```


~~pip package: ```pip install casadi-horizon```~~ *coming soon* \
~~conda package: ```conda install horizon -c ftrancesco_ruscelli```~~ *coming soon*

## Documentations
Don't forget to check the [**documentation**](https://advrhumanoids.github.io/horizon/)!  
You will obtain hands-on details about the framework: a comprehensive documentation of the project, a collection of demonstrative videos and instructions to use Horizon in Docker.

## Try it!
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/FrancescoRuscelli/horizon-live/main?urlpath=lab/tree/index.ipynb)
